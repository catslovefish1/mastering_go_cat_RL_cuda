# agents/mcts_ops.py
from __future__ import annotations

import torch
from torch import Tensor

from engine.game_state_machine import GameStateMachine
from engine.game_state import GameState
from .mcts_tree import MCTSTree


@torch.no_grad()
def run_mcts_random_root(
    tree: MCTSTree,
    root_game_state_machine: GameStateMachine,  # immutable root physics
    num_simulations: int,   # ignored for now
    komi: float = 0.0,
    debug: bool = False,
) -> Tensor:
    """
    Greedy root search using compute_scores on a GameStateMachine, fully batched:

      - Use root_game_state_machine (physics on root_state) to:
          * get legal moves at the root
          * read root tensors (boards, to_play, pass_count, zobrist_hash)
      - Collect all legal (b,a) pairs into a single big batch of size K.
      - Clone those K root states into a batched eval_game_state_machine.
      - Apply all K moves in parallel.
      - Call compute_scores() once on that eval batch.
      - Convert scores to values v in {-1,0,+1} from root player's POV.
      - Scatter back into values[b,a], and pick best action per game by argmax.

    Returns
    -------
    actions : (B,) long
        0 .. H*W-1  => board points
        H*W         => pass
    """

    # --- 1) legal mask at the root (on root_game_state_machine) ---
    legal_mask_2d = root_game_state_machine.legal_moves()  # (B, H, W) bool
    B, H, W = legal_mask_2d.shape
    assert H == W == tree.board_size, "Tree and GameStateMachine board sizes must match"

    dev = root_game_state_machine.device
    A = tree.A
    root = 0
    pass_idx = H * W

    # flatten legal board points
    flat_legal = legal_mask_2d.view(B, H * W)  # (B, H*W)
    has_legal = flat_legal.any(dim=1)          # (B,)

    # build per-action legal mask at root
    root_legal = torch.zeros((B, A), dtype=torch.bool, device=dev)
    root_legal[:, : H * W] = flat_legal
    # pass is legal only when there is no board move
    root_legal[~has_legal, pass_idx] = True

    # After: pass is always allowed as a root action
    root_legal[:, pass_idx] = True

    # store in tree for completeness
    tree.legal[:, root, :] = root_legal
    tree.P[:, root] = 1.0  # node-level prior flag (dummy)

    # --- 2) collect all legal (b,a) pairs into one batch ---
    b_idx, a_idx = root_legal.nonzero(as_tuple=True)  # (K,), (K,)
    K = b_idx.shape[0]

    # if no legal actions at all (very unlikely), everyone passes
    if K == 0:
        actions = torch.full((B,), pass_idx, dtype=torch.long, device=dev)
        tree.N[:, root] = 0.0
        if debug:
            print("[MCTS_ROOT][debug] no legal actions at all, forced pass for all games")
        return actions

    # --- 3) build batched root successor states for those K pairs ---
    # NOTE: root_game_state_machine is read-only; we clone from its tensors.
    boards_eval       = root_game_state_machine.boards[b_idx].clone()        # (K, H, W)
    to_play_eval      = root_game_state_machine.to_play[b_idx].clone()       # (K,)
    if debug:
        print("[MCTS_ROOT][debug] to_play_eval (first 3):",
              to_play_eval[0:3].cpu().tolist())
    pass_count_eval   = root_game_state_machine.pass_count[b_idx].clone()    # (K,)
    zobrist_hash_eval = root_game_state_machine.zobrist_hash[b_idx].clone()  # (K, 2)

    state_eval = GameState(
        boards=boards_eval,
        to_play=to_play_eval,
        pass_count=pass_count_eval,
        zobrist_hash=zobrist_hash_eval,
    )
    eval_game_state_machine = GameStateMachine(state_eval)

    # --- 4) build moves for each (b,a) ---
    rows = a_idx // H
    cols = a_idx % H
    moves_eval = torch.stack([rows, cols], dim=1).to(torch.long).to(dev)  # (K,2)

    # pass actions -> (-1, -1)
    is_pass = (a_idx == pass_idx)
    moves_eval[is_pass] = -1

    # apply all moves in parallel (on eval_game_state_machine only)
    eval_game_state_machine.state_transition(moves_eval)

    # --- 5) score all K successor states in parallel ---
    scores_eval = eval_game_state_machine.compute_scores(komi=komi)  # (K,2) float

    # values from ROOT player's perspective
    root_to_play_full = root_game_state_machine.to_play.to(torch.long)  # (B,)
    root_players = root_to_play_full[b_idx]                             # (K,) 0=black,1=white
    other = 1 - root_players

    if debug:
        print("[MCTS_ROOT][debug] root_engine.to_play unique:",
              torch.unique(root_game_state_machine.to_play).cpu().tolist())
        print("[MCTS_ROOT][debug] root_players unique:",
              torch.unique(root_players).cpu().tolist())
        print("[MCTS_ROOT][debug] root_players (first 32):",
              root_players[:32].cpu().tolist())

    arange_K = torch.arange(K, device=dev)
    s_root  = scores_eval[arange_K, root_players]  # (K,)
    s_other = scores_eval[arange_K, other]         # (K,)

    # v in {-1,0,1} based on who is ahead
    v_eval = torch.empty((K,), dtype=torch.float32, device=dev)
    v_eval[s_root > s_other]  =  1.0   # good for player to move
    v_eval[s_root < s_other]  = -1.0   # bad
    v_eval[s_root == s_other] =  0.0   # draw / equal

    # --- 6) scatter back into (B,A) value table ---
    values = torch.full(
        (B, A),
        -1e9,
        dtype=torch.float32,
        device=dev,
    )
    values[b_idx, a_idx] = v_eval

    # --- 7) pick best action per game (vectorized) ---
    vals_masked = values.clone()        # (B, A)
    vals_masked[~root_legal] = -1e9     # illegal -> very negative

    best_actions = torch.argmax(vals_masked, dim=1)  # (B,)

    # games with no legal actions (just in case) → forced pass
    has_any_legal = root_legal.any(dim=1)            # (B,)
    actions = best_actions.clone()
    actions[~has_any_legal] = pass_idx

    # optional: store "visit count" = #evaluated actions
    tree.N[:, root] = (values > -1e8).sum(dim=1).to(torch.float32)

    return actions
