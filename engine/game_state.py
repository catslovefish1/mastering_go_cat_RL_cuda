# engine/game_state.py
from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import Tensor

from .stones import Stone  # shared IntEnum:


@dataclass
class GameState:
    """
    Batched Go state for B parallel games.
    """

    boards: Tensor        # (B, N2) int8, values in Stone.{EMPTY,BLACK,WHITE}
    board_size: int
    to_play: Tensor       # (B,) int8 (0=black, 1=white)
    pass_count: Tensor    # (B,) int8 (0,1,2)
    zobrist_hash: Tensor  # (B, 2) int32: [:,0]=current, [:,1]=previous

    # -------- convenience properties --------

    @property
    def device(self) -> torch.device:
        return self.boards.device

    @property
    def batch_size(self) -> int:
        return int(self.boards.shape[0])

    @property
    def num_points(self) -> int:
        return int(self.boards.shape[-1])

    # -------- cloning --------

    def clone(self) -> GameState:
        """
        Deep-ish copy: new GameState object with cloned tensors
        (same values, same device, independent storage).
        """
        return GameState(
            boards=self.boards.clone(),
            board_size=self.board_size,
            to_play=self.to_play.clone(),
            pass_count=self.pass_count.clone(),
            zobrist_hash=self.zobrist_hash.clone(),
        )


def create_empty_game_state(
    batch_size: int,
    board_size: int,
    device: torch.device,
) -> GameState:
    B = batch_size
    N2 = board_size * board_size

    boards = torch.full(
        (B, N2),
        fill_value=Stone.EMPTY,   # uses shared enum
        dtype=torch.int8,
        device=device,
    )
    to_play = torch.zeros(B, dtype=torch.int8, device=device)       # BLACK to move
    pass_count = torch.zeros(B, dtype=torch.int8, device=device)
    zobrist_hash = torch.zeros(B, 2, dtype=torch.int32, device=device)

    return GameState(
        boards=boards,
        board_size=board_size,
        to_play=to_play,
        pass_count=pass_count,
        zobrist_hash=zobrist_hash,
    )
