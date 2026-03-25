# agents/mcts_ops.py
from __future__ import annotations

import torch
from torch import Tensor

from engine.board_physics import GoEnginePhysics
from .mcts_tree import MCTSTreeIndexInfo


@torch.no_grad()
def run_mcts_random_root(
    tree: MCTSTreeIndexInfo,
    engine: GoEnginePhysics,
    num_simulations: int,
) -> Tensor:
    """
    Dummy MCTS "runner" for now:

      - Ignores num_simulations.
      - Uses engine.legal_points() at the root only.
      - Fills tree.legal[:, root, :] with a per-action legal mask.
      - Samples ONE random action per game from a uniform policy over
        legal actions.
      - Pass is chosen **only** when there is no legal board move,
        matching RandomBot's behavior.

    Returns
    -------
    actions : (B,) long
        0 .. H*W-1  => board points
        H*W        => pass
    """
    # 1) get legal points at the current engine state (root state)
    legal_points = engine.legal_points()  # (B, N2) bool
    B, N2 = legal_points.shape
    assert tree.board_size * tree.board_size == N2, "Tree and engine board sizes must match"

    dev = engine.device
    A = tree.A
    root = tree.root_index
    pass_idx = N2  # last action = pass
    has_legal = legal_points.any(dim=1)                # (B,)

    # 3) build per-action legal mask at root
    root_legal = torch.zeros((B, A), dtype=torch.bool, device=dev)  # (B, A)
    root_legal[:, :N2] = legal_points                                # board points

    # ► KEY CHANGE vs your original:
    #   pass is legal ONLY for games with *no* legal placements
    root_legal[~has_legal, pass_idx] = True

    # write into tree
    tree.legal[:, root, :] = root_legal

    # 4) uniform policy over legal actions at root
    probs = root_legal.float()                             # (B, A)
    row_sums = probs.sum(dim=1, keepdim=True)              # (B, 1)
    probs = probs / row_sums.clamp(min=1.0)                # avoid div by 0

    # NOTE: P is node-level (B,M) in your current tree; we just mark
    # that the root has a non-trivial prior, but we don't store per-action P.
    tree.P[:, root] = 1.0

    # 5) sample one action per game from this root policy
    actions = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)

    # 6) very minimal stats: increment root visit count
    tree.N[:, root] += 1.0
    # leave W/Q=0 for now (no value simulation)

    return actions  # (B,)
