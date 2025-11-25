# engine/game_state.py
from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Stone:
    BLACK: int = 0
    WHITE: int = 1
    EMPTY: int = -1


@dataclass
class GoBatchState:
    """
    Batched Go state for B parallel games.
    """

    boards: Tensor        # (B, H, W) int8
    to_play: Tensor       # (B,) int8 (0=black, 1=white)
    pass_count: Tensor    # (B,) int8 (0,1,2)
    zobrist_hash: Tensor  # (B, 2) int32: [:,0]=current, [:,1]=previous

    @property
    def device(self) -> torch.device:
        return self.boards.device

    @property
    def batch_size(self) -> int:
        return int(self.boards.shape[0])

    @property
    def board_size(self) -> int:
        return int(self.boards.shape[-1])

    def to(self, device: torch.device) -> "GoBatchState":
        return GoBatchState(
            boards=self.boards.to(device),
            to_play=self.to_play.to(device),
            pass_count=self.pass_count.to(device),
            zobrist_hash=self.zobrist_hash.to(device),
        )


def create_empty_batch(
    batch_size: int,
    board_size: int,
    device: torch.device,
) -> GoBatchState:
    B = batch_size
    H = W = board_size

    boards = torch.full(
        (B, H, W),
        fill_value=Stone.EMPTY,
        dtype=torch.int8,
        device=device,
    )
    to_play = torch.zeros(B, dtype=torch.int8, device=device)       # BLACK
    pass_count = torch.zeros(B, dtype=torch.int8, device=device)
    # [:,0]=current, [:,1]=previous; start at 0
    zobrist_hash = torch.zeros(B, 2, dtype=torch.int32, device=device)

    return GoBatchState(
        boards=boards,
        to_play=to_play,
        pass_count=pass_count,
        zobrist_hash=zobrist_hash,
    )
