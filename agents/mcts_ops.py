# agents/mcts_ops.py
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor

from engine.game_state_machine import GameStateMachine
from engine.game_state import GameState
from .mcts_tree import MCTSTree


# ---------------------------------------------------------------------
# Small debug helper
# ---------------------------------------------------------------------


def _log(debug: bool, *parts) -> None:
    """Small helper to keep debug prints readable and optional."""
    if debug:
        print(*parts)


# ---------------------------------------------------------------------
# Shared action <-> move helper
# ---------------------------------------------------------------------


def actions_to_moves(actions: Tensor, board_size: int) -> Tensor:
    """
    Convert flat action indices to (row, col) moves, vectorized.

    Convention:
      - 0 .. board_size^2-1 => board points
      - board_size^2        => pass => (-1, -1)

    Parameters
    ----------
    actions : Tensor
        Tensor of any shape (...,) containing action indices.
    board_size : int
        Board size H (assuming H == W).

    Returns
    -------
    moves : Tensor
        Tensor of shape (..., 2), same leading shape as `actions`,
        dtype long, where pass actions are mapped to (-1, -1).
    """
    dev = actions.device
    H = int(board_size)
    pass_idx = H * H

    flat = actions.view(-1)  # (N,)
    rows = flat // H
    cols = flat % H

    moves = torch.empty(flat.shape[0], 2, dtype=torch.long, device=dev)
    moves[:, 0] = rows
    moves[:, 1] = cols

    is_pass = flat == pass_idx
    if is_pass.any():
        moves[is_pass] = -1

    return moves.view(*actions.shape, 2)


# ============================================================================
# HIGH-LEVEL, PAPER-LIKE API
# ============================================================================


@torch.no_grad()
def run_mcts_root(
    tree: MCTSTree,
    root_game_state_machine: GameStateMachine,
    komi: float = 0.0,
    c_puct: float = 0.1,
    debug: bool = False,
) -> Tensor:
    """
    Run a full MCTS search from the root for a batch of games.

    Conceptually, for each simulation and each (non-terminal) game b:

      1. SELECTION:
         Walk down the current search tree using the PUCT rule until we
         reach a leaf node (unexpanded / terminal / depth-limited).

      2. RECONSTRUCT LEAF POSITION:
         Rebuild the concrete Go position at the leaf as a fresh
         GameStateMachine by replaying actions from the root.

      3. EXPANSION:
         At that leaf engine state:
           - Expand it by computing the legal moves and assigning a prior P.
             (Currently: uniform prior over legal moves. Later: NN policy.)

      4. EVALUATION:
         Evaluate the leaf position to obtain a scalar value v from the
         *leaf player's* point of view (currently via game_outcomes; later NN).

      5. BACKUP:
         Propagate v back along the visited path, flipping sign at each step
         so that Q[b, node] is always from player-to-move's perspective.

    After all simulations:

      6. ROOT ACTION SELECTION:
         For each game, choose a root action based on the visit counts of
         the root's children (argmax over N).

    Returns
    -------
    actions : (B,) long
        Chosen action index for each game:
          0 .. H*H-1 => board point
          H*H        => pass
    """
    # Use legal_moves to recover (B, H, W) and sanity-check board size
    legal_mask_2d = root_game_state_machine.legal_moves()  # (B, H, W)
    B, H, W = legal_mask_2d.shape
    assert H == W == tree.board_size, "Tree and engine board sizes must match"

    # Which games are already finished at the real root?
    game_over_at_root = root_game_state_machine.is_game_over()  # (B,)

    # ----------------------------------------------------------------------
    # SELECTION → RECONSTRUCT → EXPANSION → EVALUATION → BACKUP
    # ----------------------------------------------------------------------
    num_simulations = tree.max_nodes
    for sim in range(num_simulations):
        for b in range(B):
            if bool(game_over_at_root[b].item()):
                # Real game is already over at the root position
                continue

            # Only print for game b == 0
            debug_this = bool(debug and b == 0)

            _log(
                debug_this,
                f"\n=== [SIM {sim}] [GAME {b}] ===",
            )

            # 1) SELECTION: walk from root to a leaf node
            leaf_index, path = mcts_selection_phase(
                tree=tree,
                game_index=int(b),
                c_puct=c_puct,
                debug=debug_this,
            )

            # 2) RECONSTRUCT the concrete leaf position as a fresh engine
            engine = mcts_reconstruct_leaf_position(
                tree=tree,
                root_engine=root_game_state_machine,
                game_index=int(b),
                leaf_index=leaf_index,
                debug=debug_this,
            )

            # 3) EXPANSION: legal moves + priors + terminal flag (if needed)
            mcts_expansion_phase(
                tree=tree,
                engine=engine,
                game_index=int(b),
                leaf_index=leaf_index,
                debug=debug_this,
            )

            # 4) EVALUATION: scalar v from leaf player's perspective
            value = mcts_evaluation_phase(
                engine=engine,
                komi=komi,
                debug=debug_this,
            )

            # 5) BACKUP: propagate v along the path (AlphaZero-style)
            mcts_backup_phase(
                tree=tree,
                game_index=int(b),
                path=path,
                value=value,
                debug=debug_this,
            )

            _log(
                debug_this,
                f"[MCTS] end of sim {sim}, game {b}, leaf={leaf_index}, "
                f"path_len={len(path)}, value={value}",
            )

    # ----------------------------------------------------------------------
    # ROOT ACTION SELECTION – choose an action per game from root visits
    # ----------------------------------------------------------------------
    actions = mcts_choose_action_from_root_phase(
        tree=tree,
        game_over_at_root=game_over_at_root,
        debug=debug,
    )

    return actions


# Backwards-compatible alias for the old name
run_mcts_random_root = run_mcts_root


# ============================================================================
# PHASE 1: SELECTION
# ============================================================================


def mcts_selection_phase(
    tree: MCTSTree,
    game_index: int,
    c_puct: float,
    debug: bool = False,
) -> Tuple[int, List[int]]:
    """
    Phase 1 – SELECTION:
    Starting from the root node of game `b`, repeatedly apply the PUCT rule
    to choose actions and follow children, until we reach a leaf node.

    A node is considered a leaf if:
      - it is not yet expanded, OR
      - it is marked terminal, OR
      - the depth limit is reached, OR
      - we run out of capacity to allocate a new child.

    Returns
    -------
    leaf_index : int
        Local node index in the tree for this game where we stopped.
    path : List[int]
        Sequence of node indices from root to leaf (inclusive).
    """
    b = game_index
    dev = tree.device
    H = tree.board_size
    A = tree.A

    node = int(tree.root_index[b].item())  # typically 0
    path: List[int] = [node]

    _log(debug, f"[SEL] game={b} start at root node={node}")

    reason = "unknown"

    while True:
        depth = int(tree.depth[b, node].item())

        # Stop if depth limit or terminal
        if depth >= tree.max_depth:
            reason = f"depth_limit={tree.max_depth}"
            break
        if bool(tree.is_terminal[b, node].item()):
            reason = "node_marked_terminal"
            break

        # Stop if node has never been expanded yet
        if not bool(tree.is_expanded[b, node].item()):
            reason = "unexpanded_leaf"
            break

        # We assume: expanded nodes always have a valid legal mask,
        # and pass is always legal when the game is alive.
        legal = tree.legal[b, node]  # (A,)

        child_row = tree.child_index[b, node]       # (A,)
        P_row = tree.P[b, node].to(torch.float32)   # (A,)

        N_child = torch.zeros(A, dtype=torch.float32, device=dev)
        Q_child = torch.zeros(A, dtype=torch.float32, device=dev)

        has_child = child_row >= 0
        if bool(has_child.any().item()):
            child_nodes = child_row[has_child]  # (num_children,)
            N_child[has_child] = tree.N[b, child_nodes].to(torch.float32)
            Q_child[has_child] = tree.Q[b, child_nodes].to(torch.float32)

        N_parent = tree.N[b, node].to(torch.float32)
        sqrt_N_parent = torch.sqrt(torch.clamp(N_parent, min=1.0))

        U = c_puct * P_row * sqrt_N_parent / (1.0 + N_child)
        scores = Q_child + U
        scores[~legal] = -1e9  # mask illegal moves (pass is always legal)

        if torch.all(scores <= -1e8):
            tree.is_terminal[b, node] = True
            reason = "all_scores_invalid"
            break

        a_sel = int(torch.argmax(scores).item())

        _log(
            debug,
            f"[SEL] node={node} depth={depth} N_parent={float(N_parent):.1f} "
            f"choose action a={a_sel}",
        )

        # Follow or create child
        child_n = int(child_row[a_sel].item())
        if child_n < 0:
            child_n = int(tree.next_free[b].item())
            if child_n >= tree.M:
                reason = "no_node_capacity"
                break

            # Reserve index and wire geometry
            tree.next_free[b] = child_n + 1
            tree.child_index[b, node, a_sel] = child_n

            tree.parent[b, child_n] = node
            tree.parent_action[b, child_n] = a_sel
            tree.depth[b, child_n] = depth + 1

            # Store move (row, col) from parent uniformly (pass or point)
            a_tensor = torch.tensor([a_sel], dtype=torch.long, device=dev)
            rc = actions_to_moves(a_tensor, H)[0]  # (2,)
            tree.move_pos_from_parent[b, child_n] = rc.to(torch.int8)

            # Player to move at child
            tree.to_play[b, child_n] = tree.to_play[b, node] ^ 1
            tree.is_expanded[b, child_n] = False
            tree.is_terminal[b, child_n] = False

            _log(
                debug,
                f"[SEL]   -> create new child node={child_n} from action a={a_sel}",
            )

            node = child_n
            path.append(node)
            reason = "new_child_leaf"
            break

        _log(
            debug,
            f"[SEL]   -> descend to existing child node={child_n}",
        )
        node = child_n
        path.append(node)

    leaf_index = node
    _log(
        debug,
        f"[SEL] stop at leaf node={leaf_index}, depth={int(tree.depth[b, leaf_index].item())}, "
        f"path={path}, reason={reason}",
    )
    return leaf_index, path


# ============================================================================
# PHASE 2: RECONSTRUCT LEAF POSITION
# ============================================================================


def mcts_reconstruct_leaf_position(
    tree: MCTSTree,
    root_engine: GameStateMachine,
    game_index: int,
    leaf_index: int,
    debug: bool = False,
) -> GameStateMachine:
    """
    Helper: reconstruct the concrete Go position at a leaf node as a fresh
    GameStateMachine by replaying all actions from the game's root.

    This is logically between SELECTION and (EXPANSION, EVALUATION).
    """
    b = game_index
    dev = tree.device
    H = tree.board_size

    # Recover the action sequence root -> leaf in tree space
    actions_list: List[int] = []
    node = leaf_index
    while True:
        parent = int(tree.parent[b, node].item())
        if parent < 0:
            break
        a = int(tree.parent_action[b, node].item())
        actions_list.append(a)
        node = parent
    actions_list.reverse()

    depth = int(tree.depth[b, leaf_index].item())
    _log(
        debug,
        f"[RECONSTRUCT] game={b} leaf={leaf_index} depth={depth} "
        f"replay {len(actions_list)} actions from root",
    )

    # Clone root state for game b
    boards = root_engine.boards[b : b + 1].clone()              # (1, H, W)
    to_play = root_engine.to_play[b : b + 1].clone()            # (1,)
    pass_count = root_engine.pass_count[b : b + 1].clone()      # (1,)
    zobrist_hash = root_engine.zobrist_hash[b : b + 1].clone()  # (1, 2)

    state = GameState(
        boards=boards,
        to_play=to_play,
        pass_count=pass_count,
        zobrist_hash=zobrist_hash,
    )
    engine = GameStateMachine(state)

    # Replay actions to get to leaf position
    if len(actions_list) > 0:
        a_tensor = torch.tensor(actions_list, dtype=torch.long, device=dev)  # (L,)
        moves_seq = actions_to_moves(a_tensor, H)  # (L, 2)
        for m in moves_seq:
            engine.state_transition(m.view(1, 2))

    return engine


# ============================================================================
# PHASE 3: EXPANSION
# ============================================================================


def mcts_expansion_phase(
    tree: MCTSTree,
    engine: GameStateMachine,
    game_index: int,
    leaf_index: int,
    debug: bool = False,
) -> None:
    """
    Phase 3 – EXPANSION:

      - Decide whether the leaf node should be expanded (non-terminal and
        below max depth, and not yet expanded).
      - If so, compute its legal moves and assign a prior over actions P.
      - Mark terminal nodes when appropriate.

    Invariant under this design:
      - For any non-terminal expanded node, pass is always legal.
    """
    b = game_index
    dev = tree.device
    H = tree.board_size
    A = tree.A
    pass_idx = tree.pass_idx

    depth = int(tree.depth[b, leaf_index].item())

    # Check if game is over at this leaf position
    is_over = bool(engine.is_game_over()[0].item())
    _log(
        debug,
        f"[EXPANSION] game={b} leaf={leaf_index} depth={depth} "
        f"is_over={is_over} max_depth={tree.max_depth}",
    )

    if (not is_over) and (depth < tree.max_depth) and (
        not bool(tree.is_expanded[b, leaf_index].item())
    ):
        legal_mask_2d = engine.legal_moves()    # (1, H, W) bool
        legal_flat = legal_mask_2d.view(-1)     # (H*H,)

        legal = torch.zeros(A, dtype=torch.bool, device=dev)
        legal[: H * H] = legal_flat
        legal[pass_idx] = True  # allow pass whenever the game is alive

        tree.legal[b, leaf_index] = legal

        num_legal = int(legal.sum().item())
        _log(
            debug,
            f"[EXPANSION] expand leaf={leaf_index}: num_legal={num_legal}",
        )


        prior = torch.zeros(A, dtype=torch.float32, device=dev)
        prior[legal] = 1.0 / num_legal
        tree.P[b, leaf_index] = prior.to(tree.P.dtype)
        tree.is_expanded[b, leaf_index] = True
        _log(
            debug,
            f"[EXPANSION]   set uniform prior over legal actions",
        )

    # Mark terminal if game is over or depth limit reached
    if is_over or depth >= tree.max_depth:
        tree.is_terminal[b, leaf_index] = True
        _log(
            debug,
            f"[EXPANSION] mark leaf={leaf_index} as terminal "
            f"(is_over={is_over}, depth={depth})",
        )


# ============================================================================
# PHASE 4: EVALUATION
# ============================================================================


def mcts_evaluation_phase(
    engine: GameStateMachine,
    komi: float,
    debug: bool = False,
) -> float:
    """
    Phase 4 – EVALUATION:

      - Use engine.game_outcomes(komi) to get the outcome from Black's POV.
      - Convert that into a scalar v from the *leaf player's* POV:
          * leaf_player == Black: v = outcome_black
          * leaf_player == White: v = -outcome_black
    """
    outcomes_black = engine.game_outcomes(komi=komi)  # (1,), from Black's POV
    outcome_black = int(outcomes_black[0].item())     # -1, 0, +1

    leaf_player = int(engine.to_play[0].item())       # 0 = Black, 1 = White
    if leaf_player == 0:
        v = float(outcome_black)      # leaf player is Black
    else:
        v = float(-outcome_black)     # leaf player is White

    _log(
        debug,
        f"[EVAL] outcome_black={outcome_black}, "
        f"leaf_player={leaf_player}, value={v}",
    )

    return v


# ============================================================================
# PHASE 5: BACKUP
# ============================================================================


def mcts_backup_phase(
    tree: MCTSTree,
    game_index: int,
    path: List[int],
    value: float,
    debug: bool = False,
) -> None:
    """
    Phase 5 – BACKUP (AlphaZero-style):

    Given:
      - a visited path of nodes from root to leaf (inclusive),
      - a scalar value v_leaf from the *leaf player's* perspective,

    we update N, W, and Q for every node on that path so that:

      Q[b, node] ≈ expected value from the perspective of the
      player-to-move at that node.

    Implementation:
      - Start from the leaf node with sign = +1 (leaf player's POV).
      - Walk back up the path towards the root, flipping the sign at each step.
    """
    b = game_index
    v_leaf = float(value)

    _log(
        debug,
        f"[BACKUP] game={b} path={path} leaf_value={v_leaf}",
    )

    sign = 1.0  # +v for leaf player, then flip each step towards root

    for n in reversed(path):  # iterate leaf -> ... -> root
        n_idx = int(n)
        N_old = float(tree.N[b, n_idx].item())
        W_old = float(tree.W[b, n_idx].item())

        N_new = N_old + 1.0
        W_new = W_old + sign * v_leaf
        Q_new = W_new / N_new

        tree.N[b, n_idx] = N_new
        tree.W[b, n_idx] = W_new
        tree.Q[b, n_idx] = Q_new

        _log(
            debug,
            f"[BACKUP]   node={n_idx} sign={sign:+.0f} "
            f"N:{N_old:.1f}->{N_new:.1f} "
            f"W:{W_old:.1f}->{W_new:.1f} Q={Q_new:.3f}",
        )

        sign = -sign  # flip perspective for the parent


# ============================================================================
# PHASE 6: ROOT ACTION SELECTION
# ============================================================================


def mcts_choose_action_from_root_phase(
    tree: MCTSTree,
    game_over_at_root: Tensor,
    debug: bool = False,
) -> Tensor:
    """
    Final phase – choose a root action for each game.

    For each game b:
      - Assume the root node has been expanded at least once during simulations
        (so a legal mask and priors exist), unless the game is already over.
      - For every legal root action a, read the visit count N of the
        corresponding child node (if any).
      - Select the action with highest visit count; pass is always legal.
    """
    dev = tree.device
    B = tree.B
    A = tree.A
    pass_idx = tree.pass_idx

    actions = torch.full((B,), pass_idx, dtype=torch.long, device=dev)
    root = 0  # local root index per game

    legal_root = tree.legal[:, root, :]              # (B, A)
    child_index_root = tree.child_index[:, root, :]  # (B, A)

    for b in range(B):
        if bool(game_over_at_root[b].item()):
            # If game is already over, just pass.
            actions[b] = pass_idx
            continue

        # Under our design assumptions, root should have been expanded
        # in the first simulation for any live game.
        if not bool(tree.is_expanded[b, root].item()):
            raise RuntimeError(
                f"Root node for game {b} was never expanded; "
                f"check max_nodes/max_depth or expansion logic."
            )

        legal = legal_root[b]  # (A,)

        visits = torch.zeros(A, dtype=torch.float32, device=dev)
        child_row = child_index_root[b]  # (A,)
        has_child = child_row >= 0

        if bool(has_child.any().item()):
            child_nodes = child_row[has_child]
            visits[has_child] = tree.N[b, child_nodes].to(torch.float32)

        # Mask out illegal moves (pass is guaranteed legal by construction)
        visits[~legal] = -1e9

        a_best = int(torch.argmax(visits).item())
        actions[b] = a_best

        if debug and b == 0:
            topk = torch.topk(visits, k=min(5, A))
            _log(
                True,
                f"[ROOT] game={b} choose action={a_best}, "
                f"top_visits={[(int(idx), float(val)) for val, idx in zip(topk.values, topk.indices)]}",
            )

    return actions

