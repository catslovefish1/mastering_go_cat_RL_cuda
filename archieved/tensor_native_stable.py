# board_tensor.py – GPU-optimized Go engine (delta-hash super-ko, no board clones)
# with DEBUG trace hooks for placement/removal (no logic changes)

from __future__ import annotations
import os
from typing import Optional, Dict
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
from engine import GoLegalMoveChecker as legal_module
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
    """GPU-optimized multi-game Go board with batched legal-move checking."""

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
        enable_super_ko: bool = True,
        debug_place_trace: bool = False,  # <<< DEBUG flag
    ) -> None:
        super().__init__()
        self.batch_size     = batch_size
        self.board_size     = board_size
        self.history_factor = history_factor
        self.device         = device or select_device()
        self.enable_timing  = enable_timing
        self.enable_super_ko = enable_super_ko
        self.debug_place_trace = debug_place_trace  # <<< store flag

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
        self._init_zobrist_tables()
        self._init_state()

        # Cache for legal moves and capture info
        self._last_legal_mask   = None
        self._last_info = None

    # ------------------------------------------------------------------ #
    # Static data                                                        #
    # ------------------------------------------------------------------ #
    def _init_constants(self) -> None:
        # Pre-compute as tensor for efficiency (kept for potential use)
        self.NEIGHBOR_OFFSETS = torch.tensor(
            [-self.board_size,   # N
              self.board_size,   # S
             -1,                 # W
              1],                # E
            dtype=torch.long,
            device=self.device
        )

    def _init_zobrist_tables(self) -> None:
        """Initialize Zobrist hash tables + linear helpers for super-ko."""
        if not self.enable_super_ko:
            return

        H = W = self.board_size
        self.N2 = H * W

        # Random values for each position and state: empty(0), black(1), white(2)
        torch.manual_seed(42)  # reproducible profiling
        self.zobrist_table = torch.randint(
            0, 2**31,  # int32 range; stored as int32
            (H, W, 3),
            dtype=torch.int32,
            device=self.device
        )

        # Linear helpers
        # POS_2D[r,c] -> linear index in [0, N2)
        self.POS_2D = torch.arange(self.N2, device=self.device, dtype=torch.int64).view(H, W)
        # Flattened Zobrist: (N2, 3)
        self.Zpos = self.zobrist_table.view(self.N2, 3).contiguous()
        self.ZposT = self.Zpos.transpose(0, 1).contiguous()   # <-- cache once

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
            torch.zeros(B, dtype=torch.int8, device=dev)
        )

        # Pass counter (game ends at 2)
        self.register_buffer(
            "pass_count",
            torch.zeros(B, dtype=torch.int8, device=dev)
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

        # Super-ko tracking with Zobrist hashing
        if self.enable_super_ko:
            # Current hash for each game
            self.register_buffer(
                "current_hash",
                torch.zeros(B, dtype=torch.int32, device=dev)
            )

            # History of hashes for super-ko detection
            self.register_buffer(
                "hash_history",
                torch.zeros((B, max_moves), dtype=torch.int32, device=dev)
            )

    # ==================== CORE UTILITIES ==================================== #

    def switch_player(self) -> None:
        """Switch current player and invalidate cached legal moves."""
        self.current_player = self.current_player ^ 1
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Clear cached legal moves and capture info."""
        self._last_legal_mask   = None
        self._last_info = None


    

    # ==================== LEGAL MOVES ======================================= #

    @timed_method
    def legal_moves(self) -> BoardTensor:
        """
        Compute legal moves (and capture info) and apply super-ko filtering.
        """
        if self._last_legal_mask is None:
                                                      
            legal_mask, info  = self._compute_legal_core()
            


            if self.enable_super_ko:
                legal_mask = self._filter_super_ko_vectorized(legal_mask, info)

            self._last_legal_mask   = legal_mask

            self._last_info = info
            
        return self._last_legal_mask
    
    
    @timed_method
    def _compute_legal_core(self):
        """
        Thin timed wrapper around GoLegalMoveChecker.compute_legal_moves_with_captures.
        Returns (legal_mask, info).
        """
        return self.legal_checker.compute_batch_legal_and_info(
            board=self.board,
            current_player=self.current_player,
            return_info=True,
    )

    # ======= Super-ko via delta-hash (no board clones, no .item() syncs) =======
    # no branching but extremely large size as B*N2*N2 int32 tensor
    # @timed_method
    def _filter_super_ko_vectorized(self, legal_mask: BoardTensor, info: Dict) -> BoardTensor:
        """
        Super-ko pre-filter using delta Zobrist hashes.

        For every *candidate* point (i.e., legal move), we:
          1) compute the Zobrist delta for placing a stone there,
          2) compute the Zobrist delta for removing any captured opponent stones,
          3) XOR both deltas with the current hash to get the post-move hash,
          4) reject the move if this post-move hash appears in hash_history.

        Shapes
        ------
        legal_mask : (B, H, W) bool
        capture_stone_mask : (B, N2, N2) bool  # per-move: which stones would be captured
        """ 
        B, H, W = legal_mask.shape
        N2 = H * W

        # --- Candidate mask (flattened) -----------------------------------------
        legal_flat : Tensor = legal_mask.view(B, N2)                     # (B,N2)
        cand_mask  : Tensor = legal_flat.bool()           # (B, N2)

        # --- Player / opponent lane indices -------------------------------------
        # Zpos state columns: 0=empty, 1=black, 2=white
        player = self.current_player.long()                        # (B,)
        opp    = 1 - player                                        # (B,)

        # Placement delta: EMPTY -> player
        Zpos: Tensor = self.Zpos                                  # (N2, 3)
        lin = torch.arange(N2, device=self.device).view(1, -1).expand(B, -1)   
        
        empty_col  = torch.zeros_like(player)[:, None]   # (B, 1) → broadcast to (B, N2)
        player_col = (player + 1)[:, None]               # (B, 1) → broadcast to (B, N2)
        
        place_old  = Zpos[lin, empty_col]                # (B, N2)  keys for "empty"
        place_new  = Zpos[lin, player_col]               # (B, N2)  keys for "player"
        place_delta = torch.bitwise_xor(place_old, place_new)  # (B, N2)

        # --- Capture mask from the legal checker (opponent only, diagonal excluded) ---
        cap_mask = info["capture_stone_mask"]  # (B,N2,N2)  [candidate, board-cell]
        
        # --- Capture delta: XOR all (OPPONENT -> EMPTY) toggles for captured stones ---
        # Build a per-cell toggle table D = Z_opp ^ Z_empty (one row per batch)
        # self.Zpos.T has shape (3, N2): rows are [empty, black, white]
        Z_emp: Tensor = self.ZposT[0].expand(B, -1)                    # (B, N2)
        Z_opp: Tensor = self.ZposT[(opp + 1)]                       # (B, N2)  1 or 2 per row
        D: Tensor     = torch.bitwise_xor(Z_opp, Z_emp)           # (B, N2)  per-cell (opp->empty) toggle    
        
        # Select only captured cells for each candidate; keep dense & batched
        zeros_ = torch.zeros(1, 1, N2, dtype=torch.int32, device=self.device)  # (1,1,N2)
        sel: Tensor = torch.where(cap_mask, D[:, None, :], zeros_)            # (B, N2, N2)
        
        
        # Reduce XOR along last dim → (B,N2)
        
        
        def xor_reduce_last_dim(x: Tensor) -> Tensor:
            """
            Reduce XOR along the last dimension with a power-of-two padded tree.
            Input : x ∈ ℤ^(B, N2, K)
            Output: y ∈ ℤ^(B, N2)
            """
            B_, N2_, K = x.shape

            # Pad to next power of two so each round cleanly halves the width
            P = 1 << (K - 1).bit_length()                     # next pow2 ≥ K
            if P != K:
                x = torch.cat([x, x.new_zeros(B_, N2_, P - K)], dim=2)
                K = P
 
            steps = K.bit_length() - 1
            for _ in range(steps):
                half = K // 2
                # Pairwise XOR: left half ⊕ right half
                x = torch.bitwise_xor(x[:, :, :half], x[:, :,half:K])  # (B_, N2_, half)
                K = half
            return x.squeeze(2)                                # (B, N2)
                                   # (B,N2)
        
        cap_delta : Tensor = xor_reduce_last_dim(sel) 
        
        
        

        # Candidate hashes by delta
        # new_hash[b, i] = current_hash[b] ^ place_delta[b, i] ^ cap_delta[b, i]
        new_hash = (self.current_hash[:, None] ^ place_delta) ^ cap_delta                # (B,N2)

        # # Compare against full history (mask by move_count)
        # max_moves = self.hash_history.shape[1]
        # HIST = self.hash_history                                                      # (B,max_moves)
        # hist_mask = torch.arange(max_moves, device=self.device)[None, :] < self.move_count[:, None]  # (B,max_moves)

        # # Broadcast compare: (B, N2, 1) == (B, 1, M) → (B, N2, M), masked by hist_mask
        # matches = (new_hash[:, :, None] == HIST[:, None, :]) & hist_mask[:, None, :]  # (B,N2,max_moves)
        # is_repeat_flat = matches.any(dim=2) & cand_mask                               # (B,N2)
        
        

        # # --- Final mask: keep legal but not repeating positions -----------------
        # repeat_mask = is_repeat_flat.view(B, H, W)     # (B, H, W)
        # return legal_mask & ~repeat_mask
        
        # --- Compare against full history (masked by per-row move_count), but in chunks ---
        
        #chunked version to reduce memory use on large boards
        B, N2 = new_hash.shape
        M = self.hash_history.shape[1]

        HIST = self.hash_history                              # (B, M)
        hist_mask = (torch.arange(M, device=self.device)[None, :] < self.move_count[:, None])

        # Output accumulator
        is_repeat_flat = torch.zeros(B, N2, dtype=torch.bool, device=self.device)
        
        
        HIST_CHUNK = 64  # tune: 128–1024; smaller uses less memory, larger is fewer loops

        for s in range(0, M, HIST_CHUNK):
            e = min(s + HIST_CHUNK, M)

            # Slices
            h = HIST[:, s:e]                       # (B, m)
            m = hist_mask[:, s:e]                  # (B, m) bool

            # Broadcasted equality: (B, N2, 1) == (B, 1, m) -> (B, N2, m)
            eq = (new_hash[:, :, None] == h[:, None, :])  # bool


            # Apply history mask with pure tensor ops (no if-branches)
            # (invalid columns are zeroed and will not affect the "any" reduction)
            eq = eq & m[:, None, :]
 
            # accumulate per-candidate repeats
            is_repeat_flat |= eq.any(dim=2)


        # keep only legal candidates
        is_repeat_flat &= cand_mask  # (B, N2)

        # --- Final mask: keep legal but not repeating positions ---
        repeat_mask = is_repeat_flat.view(B, H, W)
        return legal_mask & ~repeat_mask


    # ==================== MOVE EXECUTION ==================================== #

    def _update_hash_incremental(
        self, 
        b_idx: Tensor,       # (M,)  rows that actually played this ply
        rows: Tensor,        # (M,)  row indices (0..H-1) for the stone
        cols: Tensor,        # (M,)  col indices (0..W-1) for the stone
        ply: Tensor,         # (M,)  player color (0=black, 1=white)
        cap_mask_2d: Tensor  #  (M, H, W) mask of stones captured by each played move
    ) -> None: 
        """
        Incrementally update current_hash for rows in b_idx:
        XOR in the placed stone and XOR out captured opponent stones.
        """


        M = b_idx.numel()
        H = W = self.board_size
        N2 = H * W

        played_lin = self.POS_2D[rows, cols]                   # (M,)  linearized positions of played moves
        opp = (1 - ply).long()                                 # (M,)

        # --- Placement delta at the played point: EMPTY -> ply -------------------
        Zpos: Tensor = self.Zpos                # (N2, 3)  0=empty,1=black,2=white
        old_key = Zpos[played_lin, torch.zeros_like(ply)]   # (M,)  empty keys
        new_key = Zpos[played_lin, (ply + 1)]               # (M,)  player keys,+1 here means a shift from (-1,0,1) to (0,1,2)
        place_delta = torch.bitwise_xor(old_key, new_key)   # (M,)
        
         # --- Capture delta over all stones removed by this move ------------------
        cap_flat    = cap_mask_2d.view(M, N2)   # (M, N2)


        Z_emp_rows = self.ZposT[0].expand(M, -1)
        Z_opp_rows = self.ZposT[(opp + 1)]
        

        
        sel_opp = torch.where(cap_flat, Z_opp_rows, torch.zeros_like(Z_opp_rows))  # (M, N2)
        sel_emp = torch.where(cap_flat, Z_emp_rows, torch.zeros_like(Z_emp_rows))  # (M, N2)
        
        def xor_reduce_rowwise(x):      # x: (M, K)
            M, K = x.shape
            P = 1 << (K - 1).bit_length()           # next power of two ≥ K
            if P != K:
                x = torch.cat([x, x.new_zeros(M, P - K)], dim=1)  # pad with zeros
                K = P
            steps = K.bit_length() - 1
            for _ in range(steps):
                half = K // 2
                x = torch.bitwise_xor(x[:, :half], x[:, half:K])  # pairwise XOR
                K = half
            return x.squeeze(1)                                   # (M,)
        
        cap_delta_opp = xor_reduce_rowwise(sel_opp)      # (M,)
        cap_delta_emp = xor_reduce_rowwise(sel_emp)      # (M,)
        cap_delta     = torch.bitwise_xor(cap_delta_opp, cap_delta_emp)

        self.current_hash[b_idx] ^= (place_delta ^ cap_delta)

    @timed_method
    def _place_stones(self, positions: PositionTensor) -> None:
        """Vectorized stone placement, capture handling, and incremental hash update."""
        H = W = self.board_size



        # Masked indexing works even with empty selections (no need to branch)
        mask_play = (positions[:, 0] >= 0) & (positions[:, 1] >= 0)   # (B,)
        b_idx = mask_play.nonzero(as_tuple=True)[0]                   # (M,)
        rows = positions[b_idx, 0].long()
        cols = positions[b_idx, 1].long()
        ply  = self.current_player[b_idx].long()               # (M,)
        M    = b_idx.numel()

        # Get capture mask from the clean capture_stone_mask (vectorized, no loops)
        capture_stone_mask = self._last_info["capture_stone_mask"]  # (B,N2,N2),bool
        linear_idx = rows * W + cols  # Linear indices of played positions (M,)
        
        # Extract the capture masks for the specific moves using advanced indexing
        # capture_stone_mask[b_idx, linear_idx] gives us (M, N2) directly
        cap_mask = capture_stone_mask[b_idx, linear_idx]  # (M, N2)
        cap_mask_2d = cap_mask.view(M, H, W)  # (M, H, W)


        # --- Incremental hash update BEFORE mutating the board (unchanged) ---
        if self.enable_super_ko:
            self._update_hash_incremental(b_idx, rows, cols, ply, cap_mask_2d)

        # --- Place stones and apply removals on the board (UNCHANGED LOGIC) ---
        self.board[b_idx, rows, cols] = ply.to(self.board.dtype)
        self.board[b_idx] = torch.where(cap_mask_2d, Stone.EMPTY, self.board[b_idx])



    # ------------------------------------------------------------------ #
    # History                                                            #
    # ------------------------------------------------------------------ #
    @timed_method
    def _update_board_history(self) -> None:
        """
        Record the current position for every live game in the batch.
        """
        B,H,W = self.batch_size, self.board_size, self.board_size
        max_moves = self.board_history.shape[1]

        # Flatten current board state
        flat = self.board.flatten(1)  # (B, H*W)

        # Store in history if not at limit - vectorized operation
        move_idx = self.move_count.long()
        valid_mask = move_idx < max_moves
        
        b_idx = torch.arange(B, device=self.device)[valid_mask]
        mv_idx = move_idx[valid_mask]
        
        # Safe even if K==0
        self.board_history[b_idx, mv_idx] = flat[valid_mask]
        if self.enable_super_ko:
            self.hash_history[b_idx, mv_idx] = self.current_hash[valid_mask]

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
        
        # Normalize input to the board's device up front (prevents mixed-device indexing)
        positions = positions.to(self.device, non_blocking=True)

        if positions.dim() != 2 or positions.size(1) != 2:
            raise ValueError("positions must be (B, 2)")
        if positions.size(0) != self.batch_size:
            raise ValueError(f"batch size mismatch: expected {self.batch_size}, got {positions.size(0)}")

        # Record history BEFORE the move
        self._update_board_history()
        self.move_count += 1

        # Pass handling
        is_pass = ((positions[:, 0] < 0) | (positions[:, 1] < 0))
        self.pass_count = torch.where(
            is_pass,
            self.pass_count + 1,
            torch.zeros_like(self.pass_count),
        )

        # Place stones and handle captures (also updates hash incrementally)
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
