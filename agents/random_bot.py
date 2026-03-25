# agents/random_bot.py

from __future__ import annotations

import torch
from torch import Tensor

from engine.game_state_machine import GameStateMachine


class RandomBot:
    """
    Simple random Go bot.

    Uses the flat-first engine API:
      - ``legal_points()`` -> ``(B, N2)`` bool
      - Returns ``(B,)`` action IDs in ``[0, N2]`` (N2 = pass).
    """

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def select_actions(self, engine: GameStateMachine) -> Tensor:
        """
        Returns
        -------
        action_ids : (B,) long
            ``0..N2-1`` = board placement, ``N2`` = pass.
        """
        legal_points = engine.legal_points()  # (B, N2) bool
        B, N2 = legal_points.shape
        dev = engine.device

        A = N2 + 1
        legal_actions = torch.zeros(B, A, dtype=torch.bool, device=dev)
        legal_actions[:, :N2] = legal_points
        legal_actions[:, N2] = True  # pass always legal

        probs = legal_actions.float()
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1.0)

        action_ids = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)
        return action_ids
