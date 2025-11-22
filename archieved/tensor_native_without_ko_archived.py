# tensor_native.py – optimised Go engine with GoLegalMoveChecker integration
# fully int‑aligned, hot‑path free of .item() / .tolist(), and 100 % batched capture bookkeeping.

from __future__ import annotations
import os
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from collections import defaultdict

import torch
from torch import Tensor

# -----------------------------------------------------------------------------
# Shared utilities
# -----------------------------------------------------------------------------
from utils.shared import (
    select_device,
    timed_method,
    print_timing_report,
)

# -----------------------------------------------------------------------------
# GoLegalMoveChecker import
# -----------------------------------------------------------------------------
from engine import GoLegalMoveChecker_crs as legal_module
GoLegalMoveChecker = legal_module.GoLegalMoveChecker

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

# ========================= CONSTANTS =========================================

@dataclass(frozen=True)
class Stone:
    BLACK: int = 0
    WHITE: int = 1
    EMPTY: int = -1


BoardTensor    = Tensor   # (B, H, W)
PositionTensor = Tensor   # (B, 2)
PassTensor     = Tensor   # (B,)

# ========================= GO ENGINE =========================================

class TensorBoard(torch.nn.Module):
    """Vectorised multi‑game Go board with batched legal‑move checking."""

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        batch_size: int = 1,
        board_size: int = 19,
        history_factor: int = 10,
        device: Optional[torch.device] = None,
        enable_timing: bool = True,
    ) -> None:
        super().__init__()
        self.batch_size     = batch_size
        self.board_size     = board_size
        self.history_factor = history_factor
        self.device         = device or select_device()
        self.enable_timing  = enable_timing

        # Timing infrastructure
        if enable_timing:
            self.timings = defaultdict(list)
            self.call_counts = defaultdict(int)
        else:
            self.timings = {}
            self.call_counts = {}

        # Core legal move checker
        self.legal_checker = GoLegalMoveChecker(
            board_size=board_size,
            device=self.device
        )

        self._init_constants()
        self._init_state()

        # Cache for legal moves and capture info
        self._last_legal_mask   = None
        self._last_capture_info = None

    # ------------------------------------------------------------------ #
    # Static data                                                        #
    # ------------------------------------------------------------------ #
    def _init_constants(self) -> None:
        # Pre-compute as tensor for efficiency
        self.NEIGHBOR_OFFSETS = torch.tensor(
            [-self.board_size,   # N
              self.board_size,   # S
             -1,                 # W
              1],                # E
            dtype=torch.long,
            device=self.device
        )

    # ------------------------------------------------------------------ #
    # Mutable state                                                      #
    # ------------------------------------------------------------------ #
    def _init_state(self) -> None:
        B, H, W = self.batch_size, self.board_size, self.board_size
        dev     = self.device

        # Main board state
        self.register_buffer(
            "board",
            torch.full((B, H, W), Stone.EMPTY, dtype=torch.int8, device=dev)
        )
        
        # Current player (0=black, 1=white)
        self.register_buffer(
            "current_player",
            torch.zeros(B, dtype=torch.uint8, device=dev)
        )
        
        # Pass counter (game ends at 2)
        self.register_buffer(
            "pass_count",
            torch.zeros(B, dtype=torch.uint8, device=dev)
        )

        # History tracking
        max_moves = H * W * self.history_factor
        self.register_buffer(
            "board_history",
            torch.full((B, max_moves, H * W), -1, dtype=torch.int8, device=dev)
        )
        self.register_buffer(
            "move_count",
            torch.zeros(B, dtype=torch.int16, device=dev)
        )
        
        
    

    # ==================== CORE UTILITIES ==================================== #
    
    def switch_player(self) -> None:
        """Switch current player and invalidate cached legal moves."""
        self.current_player = self.current_player ^ 1
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Clear cached legal moves and capture info."""
        self._last_legal_mask   = None
        self._last_capture_info = None

    # ==================== LEGAL MOVES ======================================= #
    
    @timed_method
    def legal_moves(self) -> BoardTensor:
        """
        Get legal moves for current position.
        Caches both legal moves and capture info for efficiency.
        """
        if self._last_legal_mask is None:
            legal_mask, cap_info = self.legal_checker.compute_legal_moves_with_captures(
                board=self.board,
                current_player=self.current_player,
                return_capture_info=True,
            )
            self._last_legal_mask   = legal_mask
            self._last_capture_info = cap_info
        
        return self._last_legal_mask

    # ==================== MOVE EXECUTION ==================================== #

    @timed_method
    def _place_stones(self, positions: PositionTensor) -> None:
        
        """Fully vectorized stone placement and capture handling."""
        H = W = self.board_size
        
        # Check for all passes
        mask_play = (positions[:, 0] >= 0) & (positions[:, 1] >= 0)
        if not mask_play.any():
            return
        
        # ------------------------------------------------------------------ #
        # 1) Place stones                                                    #
        # ------------------------------------------------------------------ #
        b_idx = mask_play.nonzero(as_tuple=True)[0]
        rows = positions[b_idx, 0].long()
        cols = positions[b_idx, 1].long()
        ply = self.current_player[b_idx].long()
        self.board[b_idx, rows, cols] = ply.to(self.board.dtype)
        
        # ------------------------------------------------------------------ #
        # 2) Vectorized capture removal                                      #
        # ------------------------------------------------------------------ #
        roots = self._last_capture_info["roots"]           # (B, N²)
        cap_groups = self._last_capture_info["capture_groups"]  # (B,H,W,4)
        
        cap_sizes = self._last_capture_info["capture_sizes"]    # (B,H,W,4)
        total_caps = self._last_capture_info["total_captures"]  # (B,H,W)
        
        # Get groups to capture at played positions
        groups_at_moves = cap_groups[b_idx, rows, cols]  # (M, 4)
        
        # Build capture mask for all games at once
        # Expand dimensions for broadcasting
        roots_selected = roots[b_idx]  # (M, N²)
        groups_expanded = groups_at_moves.view(len(b_idx), 1, 4)  # (M, 1, 4)
        roots_expanded = roots_selected.view(len(b_idx), -1, 1)   # (M, N², 1)
        
        # Check which positions belong to groups that should be captured
        # This creates a (M, N², 4) tensor, then reduces to (M, N²)
        is_captured_group = (roots_expanded == groups_expanded) & (groups_expanded >= 0)
        capture_mask = is_captured_group.any(dim=2)  # (M, N²)
        
        # Reshape and apply removal
        capture_mask_2d = capture_mask.view(len(b_idx), H, W)
        self.board[b_idx] = torch.where(
            capture_mask_2d,
            Stone.EMPTY,
            self.board[b_idx]
        )

    # ------------------------------------------------------------------ #
    # History                                                            #
    # ------------------------------------------------------------------ #
    @timed_method
    def _update_board_history(self) -> None:
        """
        Record the current position for every live game in the batch.
        
        board_history  : (B, max_moves, H*W)  int8   – -1 empty, 0 black, 1 white
        move_count[b]  : how many moves have already been written for board b
        """
        B, H, W = self.batch_size, self.board_size, self.board_size
        max_moves = self.board_history.shape[1]

        # Flatten current board state
        flat = self.board.flatten(1)  # (B, H*W)
        
        # Store in history if not at limit
        move_idx = self.move_count.long()
        valid = move_idx < max_moves
        
        if valid.any():
            b_idx = torch.arange(B, device=self.device)[valid]
            mv_idx = move_idx[valid]
            self.board_history[b_idx, mv_idx] = flat[b_idx]

    # ------------------------------------------------------------------ #
    # Game loop                                                          #
    # ------------------------------------------------------------------ #
    @timed_method
    def step(self, positions: PositionTensor) -> None:
        """
        Execute one move for each game in the batch.
        
        Parameters
        ----------
        positions : (B, 2) tensor
            Row, column coordinates. Negative values indicate pass.
        """
        if positions.dim() != 2 or positions.size(1) != 2:
            raise ValueError("positions must be (B, 2)")
        if positions.size(0) != self.batch_size:
            raise ValueError(f"batch size mismatch: expected {self.batch_size}, got {positions.size(0)}")

        # Record history before move
        self._update_board_history()
        self.move_count += 1

        # Handle passes
        is_pass = (positions[:, 0] < 0) | (positions[:, 1] < 0)
        self.pass_count = torch.where(
            is_pass,
            self.pass_count + 1,
            torch.zeros_like(self.pass_count),
        )

        # Place stones and handle captures
        self._place_stones(positions)
        
        # Switch to next player
        self.switch_player()

    # ------------------------------------------------------------------ #
    # Game state                                                         #
    # ------------------------------------------------------------------ #
    def is_game_over(self) -> PassTensor:
        """Check if any games have ended (2 consecutive passes)."""
        return self.pass_count >= 2

    def compute_scores(self) -> Tensor:
        """
        Compute simple territory scores (stone count only).
        
        Returns
        -------
        scores : (B, 2) tensor
            Black and white stone counts for each game.
        """
        black = (self.board == Stone.BLACK).sum((1, 2)).float()
        white = (self.board == Stone.WHITE).sum((1, 2)).float()
        return torch.stack([black, white], dim=1)

    # ------------------------------------------------------------------ #
    # Timing                                                             #
    # ------------------------------------------------------------------ #
    def print_timing_report(self, top_n: int = 30) -> None:
        """Print timing statistics if timing is enabled."""
        if self.enable_timing:
            print_timing_report(self, top_n)
