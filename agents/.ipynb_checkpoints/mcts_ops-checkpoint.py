# agents/mcts_ops.py
from __future__ import annotations

import torch
from torch import Tensor

from engine.board_physics import GoEnginePhysics
from engine.board_state import GoBatchState
from .mcts_tree import MCTSTreeIndexInfo


@torch.no_grad()
def run_mcts_random_root(
    tree: MCTSTreeIndexInfo,
    engine: GoEnginePhysics,
    num_simulations: int,   # ignored for now
    komi: float = 0.0,
    debug: bool = False,
) -> Tensor:
    """
    Greedy root search using compute_scores, but fully batched:

      - Build legal actions at root for all games.
      - Collect all legal (b,a) pairs into a single big batch of size K.
      - Clone those K root states into a batched GoEnginePhysics.
      - Apply all K moves in parallel.
      - Call compute_scores() once on that batch.
      - Convert scores to values v in {-1,0,+1} from root player's POV.
      - Scatter back into values[b,a], and pick best action per game by argmax.

    Returns
    -------
    actions : (B,) long
        0 .. H*W-1  => board points
        H*W        => pass
    """
    # --- 1) legal mask at the root (on the VIRTUAL engine) ---
    legal_mask_2d = engine.legal_moves()  # (B, H, W) bool
    B, H, W = legal_mask_2d.shape
    assert H == W == tree.board_size, "Tree and engine board sizes must match"

    dev = engine.device
    A = tree.A
    root = tree.root_index
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

    # --- 3) build batched root states for those K pairs ---
    boards_eval       = engine.boards[b_idx].clone()        # (K, H, W)
    to_play_eval      = engine.to_play[b_idx].clone()       # (K,)
    pass_count_eval   = engine.pass_count[b_idx].clone()    # (K,)
    zobrist_hash_eval = engine.zobrist_hash[b_idx].clone()  # (K, 2)

    state_eval = GoBatchState(
        boards=boards_eval,
        to_play=to_play_eval,
        pass_count=pass_count_eval,
        zobrist_hash=zobrist_hash_eval,
    )
    engine_eval = GoEnginePhysics(state_eval)

    # --- 4) build moves for each (b,a) ---
    rows = a_idx // H
    cols = a_idx % H
    moves_eval = torch.stack([rows, cols], dim=1).to(torch.long).to(dev)  # (K,2)

    # pass actions -> (-1, -1)
    is_pass = (a_idx == pass_idx)
    moves_eval[is_pass] = -1

    # apply all moves in parallel
    engine_eval.state_transition(moves_eval)

    # --- 5) score all K successor states in parallel ---
    scores_eval = engine_eval.compute_scores(komi=komi)  # (K,2) float

    # values from root player's perspective
    root_players = to_play_eval.to(torch.long)  # (K,) 0=black,1=white
    other = 1 - root_players

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

    # --- 8) debug block: global and per-game winning stats ---
    if debug:
        win_mask  = (values ==  1.0) & root_legal   # only legal + winning
        draw_mask = (values ==  0.0) & root_legal
        loss_mask = (values == -1.0) & root_legal

        total_wins  = int(win_mask.sum().item())
        total_draws = int(draw_mask.sum().item())
        total_loss  = int(loss_mask.sum().item())

        print(
            f"[MCTS_ROOT][debug] B={B}, H={H}, W={W}, A={A}, K={K}",
            flush=True,
        )
        print(
            f"[MCTS_ROOT][debug] winning successors (global): "
            f"{total_wins}, draw: {total_draws}, loss: {total_loss}",
            flush=True,
        )

        # Per-game stats: how many winning actions per b
        win_counts_per_b = win_mask.sum(dim=1)  # (B,)
        num_with_win = int((win_counts_per_b > 0).sum().item())
        num_no_win   = B - num_with_win

        print(
            f"[MCTS_ROOT][debug] games with ≥1 winning action: {num_with_win} / {B}",
            flush=True,
        )
        print(
            f"[MCTS_ROOT][debug] games with 0 winning actions: {num_no_win}",
            flush=True,
        )

        # For *every* game with ≥1 winning action, show ONE example (to keep logs small)
        for b in range(B):
            wc = int(win_counts_per_b[b].item())
            if wc == 0:
                continue

            win_actions_b = win_mask[b].nonzero(as_tuple=True)[0]  # (k_b,)
            a0 = int(win_actions_b[0].item())

            if a0 == pass_idx:
                r_str, c_str = "-1", "-1"
            else:
                r0 = a0 // H
                c0 = a0 % H
                r_str, c_str = str(int(r0)), str(int(c0))

            print(
                f"[MCTS_ROOT][debug] game b={b}, {wc} winning actions (show 1):",
                flush=True,
            )
            print(
                f"  a={a0:3d} (r={r_str}, c={c_str}) winning",
                flush=True,
            )

    return actions


