# agents/random_bot.py

from __future__ import annotations
from typing import Optional  # you can actually delete this now

import torch
from torch import Tensor

from engine.game_state_machine import GameStateMachine


class RandomBot:
    """
    Go bot that works directly with GameStateMachine:
    - Asks engine.legal_moves() for a (B, H, W) legal mask.
    - Samples one legal move uniformly per game.
    - Returns [-1, -1] when no legal moves (pass).
    """

    def __init__(self) -> None:
        # No device stored; always use engine.device
        pass

    @torch.no_grad()
    def select_moves(self, game_state_machine: GameStateMachine) -> Tensor:
        """
        Args
        ----
        engine : GameStateMachine
            The engine holding the current batched state.

        Returns
        -------
        moves : (B,2) long
            (row, col) per game; (-1, -1) for pass.
        """
        # 1) get legal mask from engine; also primes internal caches
        legal_mask = game_state_machine.legal_moves()   # (B,H,W) bool
        B, H, W = legal_mask.shape
        dev = game_state_machine.device                 # inherit device from state / engine

        # 2) start with all passes
        moves = torch.full((B, 2), -1, dtype=torch.long, device=dev)

        # 3) find which games have at least one legal move
        flat_legal = legal_mask.view(B, -1)       # (B, H*W)
        has_legal = flat_legal.any(dim=1)         # (B,)
        if not has_legal.any():
            return moves  # everyone passes

        playable_idx = has_legal.nonzero(as_tuple=True)[0]  # (B_play,)
        playable_legal = flat_legal[playable_idx]           # (B_play, H*W)

        # 4) uniform distribution over legal points
        probs = playable_legal.float()
        row_sums = probs.sum(dim=1, keepdim=True)           # (B_play, 1)
        probs = probs / row_sums.clamp(min=1.0)             # defensive

        # 5) sample one legal flat index per playable game
        flat_indices = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B_play,)
        rows = flat_indices // W
        cols = flat_indices % W

        # 6) write them back into the (B,2) moves tensor
        moves[playable_idx] = torch.stack([rows, cols], dim=1)
        return moves


