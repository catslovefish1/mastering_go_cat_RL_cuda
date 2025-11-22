"""basic.py - Tensor-native Go agent with random rollouts (minimal)."""

from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor

from engine.board_tensor import TensorBoard


def select_device() -> torch.device:
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TensorBatchBot:
    """Fully vectorized Go bot using uniform random move selection."""

    def __init__(self, device: Optional[torch.device | str] = None):
        """Initialize bot with specified or auto-selected device."""
        self.device = torch.device(device) if device is not None else select_device()

    def select_moves(self, boards: TensorBoard) -> Tensor:
        """Select moves for all games using a uniform random policy.

        Args:
            boards: TensorBoard instance with batched Go games.

        Returns:
            Tensor of shape (B, 2) with [row, col] coordinates (int32).
            Returns [-1, -1] for pass moves.
        """
        # Legal moves mask: (B, H, W) bool
        legal_moves = boards.legal_moves()
        batch_size, _, width = legal_moves.shape

        # 1) Start with all passes: [-1, -1] for every game
        moves = torch.full((batch_size, 2), -1, dtype=torch.long, device=self.device)


        # 2) Find games that actually have at least one legal move
        games_with_moves = legal_moves.any(dim=(1, 2))  # (B,)
        if not games_with_moves.any():
            return moves  # everyone passes

        # 3) Take only playable games and flatten board positions to (B_play, H*W)
        playable_legal = legal_moves[games_with_moves]          # (B_play, H, W)
        flat_legal = playable_legal.view(playable_legal.shape[0], -1)  # (B_play, H*W)

        # 4) Build uniform probabilities over legal positions
        probs = flat_legal.float()
        row_sums = probs.sum(dim=1, keepdim=True)           # (B_play, 1)
        probs = probs / row_sums.clamp(min=1.0)             # avoid div by 0 (defensive)

        # 5) Sample one flat index per playable game
        flat_indices = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B_play,)

        # 6) Convert flat index -> (row, col) via integer arithmetic
        rows = flat_indices // width
        cols = flat_indices % width

        # 7) Write sampled moves back into the full (B, 2) array
        moves[games_with_moves] = torch.stack([rows, cols], dim=1)

        return moves

