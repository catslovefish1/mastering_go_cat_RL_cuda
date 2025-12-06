# agents/random_bot.py

from __future__ import annotations

import torch
from torch import Tensor

from engine.game_state_machine import GameStateMachine
from .mcts_ops import actions_to_moves  # reuse the shared helper


class RandomBot:
    """
    Simple random Go bot:

    - Asks engine.legal_moves() for a (B, H, W) mask of legal board points.
    - Extends that to an action space of size A = H*H + 1 by treating
      the last index as "pass", which is always legal.
    - Samples one action uniformly over all legal actions (including pass).
    - Converts the sampled action indices to (row, col) moves:
        * 0 .. H*H-1 -> board points
        * H*H        -> pass -> (-1, -1)
    """

    def __init__(self) -> None:
        # No device stored; we always take it from the engine
        pass

    @torch.no_grad()
    def select_moves(self, game_state_machine: GameStateMachine) -> Tensor:
        """
        Args
        ----
        game_state_machine : GameStateMachine
            Engine holding the current batched state.

        Returns
        -------
        moves : (B, 2) long
            (row, col) per game; (-1, -1) encodes pass.
        """
        # 1) Legal board points
        legal_mask_2d = game_state_machine.legal_moves()  # (B, H, W) bool
        B, H, W = legal_mask_2d.shape
        dev = game_state_machine.device

        # 2) Build extended legal mask over actions: A = H*H + 1
        A = H * H + 1
        pass_idx = A - 1

        flat_board = legal_mask_2d.view(B, H * H)              # (B, H*H)
        legal_actions = torch.zeros(B, A, dtype=torch.bool, device=dev)
        legal_actions[:, : H * H] = flat_board
        legal_actions[:, pass_idx] = True                      # pass always legal

        # 3) Turn legal mask into uniform probabilities over legal actions
        probs = legal_actions.float()                          # (B, A)
        row_sums = probs.sum(dim=1, keepdim=True)              # (B, 1)
        # Defensive clamp; in practice row_sums >= 1 because pass is legal
        probs = probs / row_sums.clamp(min=1.0)

        # 4) Sample one action index per game
        actions = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)

        # 5) Convert action indices -> (row, col) moves (handles pass)
        moves = actions_to_moves(actions, board_size=H)  # (B, 2)
        return moves
