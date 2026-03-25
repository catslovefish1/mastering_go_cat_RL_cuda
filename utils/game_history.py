# utils/game_history.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


@dataclass
class GameHistory:
    """Container for a batch of Go games over time (batch-first)."""

    # ---- shape / config ----
    T_max: int
    B_tracked: int
    H: int
    device: torch.device

    # ---- main buffers (allocated in __post_init__) ----
    boards: Tensor = field(init=False)   # (B_tracked, T_max+1, N2)
    to_play: Tensor = field(init=False)  # (B_tracked, T_max+1)
    hashes: Tensor = field(init=False)   # (B_tracked, T_max+1)
    action_ids: Tensor = field(init=False)   # (B_tracked, T_max)

    # ---- final info (set in finalize) ----
    finished: Optional[Tensor] = None    # (B_tracked,)
    scores:   Optional[Tensor] = None    # (B_tracked, 2)
    win_or_loss: Optional[Tensor] = None #(B_tracked, )

    # How many plies are actually valid (0..T_max)
    T_actual: int = 0

    # ---------- allocator (runs after __init__) ----------

    def __post_init__(self) -> None:
        """Allocate tensors once when the object is created."""
        B = self.B_tracked
        T = self.T_max
        H = self.H
        N2 = H * H
        d = self.device

        self.boards = torch.empty((B, T + 1, N2), dtype=torch.int8,  device=d)
        self.to_play = torch.empty((B, T + 1),      dtype=torch.int8,  device=d)
        self.hashes = torch.empty((B, T + 1),       dtype=torch.int32, device=d)
        self.action_ids  = torch.empty((B, T),      dtype=torch.int64, device=d)

    # ---------- in-place finalize ----------

    def finalize(self, num_plies: int, finished: Tensor, scores: Tensor, win_or_loss: Tensor) -> "GameHistory":
        """Record T_actual, finished, and scores without changing tensor storage."""
        self.T_actual = int(num_plies)
        self.finished = finished
        self.scores   = scores
        self.win_or_loss = win_or_loss
        return self
