# board_tensor.py – GPU-optimized Go engine (delta-hash super-ko, CSR captures)
# Uses CSR (cap_indptr/cap_indices) both for super-ko AND for in-place removals.

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
    """GPU-optimized multi-game Go board with batched legal-move checking."""

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        batch_size: int = 1,
        board_size: int = 19,
        history_factor: int = 1,
        device: Optional[torch.device] = None,
        enable_timing: bool = True,
        enable_super_ko: bool = True,
        debug_place_trace: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size     = batch_size
        self.board_size     = board_size
        self.history_factor = history_factor
        self.device         = device or select_device()
        self.enable_timing  = enable_timing
        self.enable_super_ko = enable_super_ko
        self.debug_place_trace = debug_place_trace

        # Timing infrastructure
        if enable_timing:
            self.timings = defaultdict(list)
            self.call_counts = defaultdict(int)
        else:
            self.timings = {}
            self.call_counts = {}

        # Core legal move checker (emits CSR + legacy dense for migration)
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
        self.POS_2D = torch.arange(self.N2, device=self.device, dtype=torch.int64).view(H, W)
        self.Zpos = self.zobrist_table.view(self.N2, 3).contiguous()      # (N2,3)
        self.ZposT = self.Zpos.transpose(0, 1).contiguous()               # (3,N2)

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
            torch.full((16, max_moves, H * W), -1, dtype=torch.int8, device=dev)
        )
        self.register_buffer(
            "move_count",
            torch.zeros(B, dtype=torch.int16, device=dev)
        )

        # Super-ko tracking with Zobrist hashing
        if self.enable_super_ko:
            self.register_buffer(
                "current_hash",
                torch.zeros(B, dtype=torch.int32, device=dev)
            )
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
            legal_mask, info = self._compute_legal_core()
            if self.enable_super_ko:
                legal_mask = self._filter_super_ko_vectorized(legal_mask, info)
            self._last_legal_mask = legal_mask
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
        
    # ======= Super-ko via delta-hash (CSR captures, no clones, no .item() syncs) =======
    # @timed_method
    # def _filter_super_ko_vectorized(self, legal_mask: BoardTensor, info: Dict) -> BoardTensor:
    #     """
    #     Use Zobrist deltas + CSR capture lists to remove super-ko repeats.
    #     CSR inputs (flat row-space):
    #       cap_indptr:  (B*N2 + 1,) int32
    #       cap_indices: (L,)        int32
    #     Row id: r = b*N2 + i
    #     """
    #     B, H, W = legal_mask.shape
    #     N2 = H * W
    #     dev = self.device

    #     # --- Candidate mask (flattened) -----------------------------------------
    #     legal_flat: Tensor = legal_mask.view(B, N2).bool()          # (B, N2)
    #     cand_mask: Tensor  = legal_flat                              # (B, N2)

    #     # --- Player / opponent lanes --------------------------------------------
    #     player = self.current_player.long()                          # (B,)
    #     opp    = 1 - player

    #     # --- Placement delta: EMPTY -> player -----------------------------------
    #     Zpos: Tensor = self.Zpos                                     # (N2, 3) int32
    #     lin = torch.arange(N2, device=dev).view(1, -1).expand(B, -1) # (B, N2)
    #     empty_col  = torch.zeros_like(player)[:, None]               # (B, 1)
    #     player_col = (player + 1)[:, None]                           # (B, 1)
    #     place_old  = Zpos[lin, empty_col]                            # (B, N2) int32
    #     place_new  = Zpos[lin, player_col]                           # (B, N2) int32
    #     place_delta = torch.bitwise_xor(place_old, place_new)        # (B, N2) int32

    #     # Per-cell toggle for opponent→empty (used for captured stones)
    #     Z_emp: Tensor = self.ZposT[0].expand(B, -1)                  # (B, N2) int32
    #     Z_opp: Tensor = self.ZposT[(opp + 1)]                        # (B, N2) int32
    #     D: Tensor     = torch.bitwise_xor(Z_opp, Z_emp)              # (B, N2) int32

    #     # Start with placement only
    #     new_hash = (self.current_hash[:, None] ^ place_delta).contiguous()  # (B, N2) int32

    #     # --- Capture handling via CSR -------------------------------------------
    #     cap_indptr  = info["cap_indptr"].reshape(-1).to(torch.int64)   # (B*N2 + 1,)
    #     cap_indices = info["cap_indices"].reshape(-1).to(torch.int64)  # (L,)
    #     cand_cap = (legal_flat & info["can_capture_any"]).reshape(-1)  # (B*N2,)

    #     starts = cap_indptr[:-1]                                       # (B*N2,)
    #     lens   = (cap_indptr[1:] - cap_indptr[:-1])                    # (B*N2,)
    #     keep   = cand_cap & (lens > 0)                                 # (B*N2,)

    #     if keep.any():
    #         seg_ids = keep.nonzero(as_tuple=True)[0]                   # (T,)
    #         b_seg = (seg_ids // N2).to(torch.int64)                    # (T,)
    #         i_seg = (seg_ids %  N2).to(torch.int64)                    # (T,)
    #         s_seg = starts[seg_ids]                                    # (T,)
    #         n_seg = lens[seg_ids]                                      # (T,) int64

    #         # Pad to the max segment length so we can do one dense gather + XOR reduce
    #         Lmax = int(n_seg.max().item())                             # small host read
    #         offs = torch.arange(Lmax, device=dev, dtype=torch.int64)   # (Lmax,)
    #         idx  = s_seg[:, None] + offs[None, :]                      # (T, Lmax)
    #         mask = offs[None, :] < n_seg[:, None]                      # (T, Lmax)

    #         # Captured stone indices per (b,i), padded
    #         jmat = torch.where(mask, cap_indices[idx], torch.zeros_like(idx))      # (T, Lmax)

    #         # Zobrist toggles for those stones and XOR-reduce across width
    #         vals = D[b_seg].gather(1, jmat)                                        # (T, Lmax)
    #         vals = torch.where(mask, vals, vals.new_zeros((), dtype=vals.dtype))   # zero pads

    #         # Tree XOR reduce along last dim
    #         acc = vals
    #         width = Lmax
    #         while width > 1:
    #             half = width // 2
    #             acc = torch.bitwise_xor(acc[:, :half], acc[:, half:half*2])
    #             if width & 1:
    #                 acc = torch.bitwise_xor(acc, acc[:, -1:])
    #             width = acc.size(1)
    #         cap_delta = acc.squeeze(1)                                             # (T,)

    #         # Blend captured-stone delta into the post-move hash for those rows
    #         new_hash[b_seg, i_seg] ^= cap_delta

    #     # ---- Compare against history in chunks ---------------------------------
    #     M = self.hash_history.shape[1]
    #     hist_mask = (torch.arange(M, device=dev)[None, :] < self.move_count[:, None])  # (B, M) bool

    #     is_repeat_flat = torch.zeros(B, N2, dtype=torch.bool, device=dev)
    #     HIST_CHUNK = 32
    #     for s in range(0, M, HIST_CHUNK):
    #         e = min(s + HIST_CHUNK, M)
    #         h = self.hash_history[:, s:e]                           # (B, m)
    #         m = hist_mask[:, s:e]                                   # (B, m) bool
    #         eq = (new_hash[:, :, None] == h[:, None, :])            # (B, N2, m)
    #         eq &= m[:, None, :]                                     # mask invalid columns
    #         is_repeat_flat |= eq.any(dim=2)

    #     # Keep only legal candidates that are not repeats
    #     is_repeat_flat &= legal_flat
    #     repeat_mask = is_repeat_flat.view(B, H, W)
    #     return legal_mask & ~repeat_mask

    # ======= Super-ko via delta-hash (CSR captures, no clones, no .item() syncs) =======
    @timed_method
    def _filter_super_ko_vectorized(self, legal_mask: BoardTensor, info: Dict) -> BoardTensor:
        """
        Use Zobrist deltas + CSR capture lists to remove super-ko repeats.
        CSR inputs (flat row-space):
          cap_indptr:  (B*N2 + 1,) int32
          cap_indices: (L,)        int32
        Row id: r = b*N2 + i
        """
        B, H, W = legal_mask.shape
        N2 = H * W
        dev = self.device

        # --- Candidate mask (flattened) -----------------------------------------
        legal_flat: Tensor = legal_mask.view(B, N2).bool()          # (B, N2)

        # --- Player / opponent lanes --------------------------------------------
        player = self.current_player.long()                        # (B,)
        opp    = 1 - player

        z_empty    = self.ZposT[0]                                 # (N2,) int32
        z_by_color = self.ZposT[1:3]                               # (2,N2) int32
        z_place    = z_by_color[player]                            # (B,N2) int32
        place_delta = z_empty ^ z_place                            # (B,N2) int32

        Z_emp = self.ZposT[0].expand(B, -1)                        # (B,N2) view
        Z_opp = self.ZposT[(opp + 1)]                              # (B,N2) gather-by-lane
        D     = Z_opp ^ Z_emp                                      # (B,N2) int32

        new_hash = (self.current_hash[:, None] ^ place_delta).contiguous()  # (B,N2)

        # --- Capture handling via CSR (TILED; no (T, Lmax) pad) ---
        cap_indptr  = info["cap_indptr"].reshape(-1).to(torch.int64)  # (B*N2 + 1,)
        cap_indices = info["cap_indices"].reshape(-1).to(torch.int64) # (L,)
        cand_cap = (legal_flat & info["can_capture_any"]).reshape(-1) # (B*N2,)

        starts = cap_indptr[:-1]                                      # (B*N2,)
        lens   = (cap_indptr[1:] - cap_indptr[:-1])                   # (B*N2,)
        keep   = cand_cap & (lens > 0)

        if keep.any():
            seg_ids = keep.nonzero(as_tuple=True)[0]                  # (T,)
            b_seg = (seg_ids // N2).to(torch.int64)                   # (T,)
            i_seg = (seg_ids %  N2).to(torch.int64)                   # (T,)
            s_seg = starts[seg_ids]                                   # (T,)
            n_seg = lens[seg_ids]                                     # (T,) int64

            # Tiled reduction: memory O(T * TILE) instead of O(T * Lmax)
            TILE = 64  # tune: 32/64/128 depending on your GPU
            cap_delta = torch.zeros(seg_ids.numel(), dtype=torch.int32, device=dev)

            max_len = int(n_seg.max().item())                         # small host read (ok)
            offs = torch.arange(TILE, device=dev, dtype=torch.int64)  # (TILE,)

            for start in range(0, max_len, TILE):
                rel  = start + offs                                   # (TILE,)
                # (T,TILE) mask for rows shorter than current tile
                valid = rel[None, :] < n_seg[:, None]

                # (T,TILE) stone indices; safe to compute even where invalid (we'll mask)
                idx = s_seg[:, None] + rel[None, :]
                j   = cap_indices[idx.clamp_max(cap_indices.numel() - 1)]
                j   = torch.where(valid, j, j.new_zeros(()))          # zero fill

                # gather toggles and XOR-reduce along tile width
                d = D[b_seg].gather(1, j)
                d = torch.where(valid, d, d.new_zeros(()))

                # pairwise XOR tree on the (T,TILE) block
                acc = d
                width = acc.size(1)
                while width > 1:
                    half = width // 2
                    acc = acc[:, :half] ^ acc[:, half:half*2]
                    if width & 1:
                        acc = acc ^ acc[:, -1:]
                    width = acc.size(1)

                cap_delta ^= acc.squeeze(1)                            # (T,)

            new_hash[b_seg, i_seg] ^= cap_delta

        # --- Super-ko check: use sort+search (no 3D (B,N2,m) boolean) ---
        repeat_mask = self._repeat_mask_from_history(new_hash, legal_mask)
        return legal_mask & ~repeat_mask

    # ==================== MOVE EXECUTION ==================================== #
    @timed_method
    def _update_hash_incremental(
        self, 
        b_idx: Tensor,       # (M,)  rows that actually played this ply
        rows: Tensor,        # (M,)  row indices (0..H-1) for the stone
        cols: Tensor,        # (M,)  col indices (0..W-1) for the stone
        ply: Tensor,         # (M,)  player color (0=black, 1=white)
        cap_mask_2d: Tensor  # (M, H, W) mask of stones captured by each played move
    ) -> None: 
        """
        Incrementally update current_hash for rows in b_idx:
        XOR in the placed stone and XOR out captured opponent stones.
        """
        M = b_idx.numel()
        H = W = self.board_size
        N2 = H * W

        played_lin = self.POS_2D[rows, cols]                   # (M,)
        opp = (1 - ply).long()                                 # (M,)

        # Placement delta at the played point: EMPTY -> ply
        Zpos: Tensor = self.Zpos                # (N2, 3)
        old_key = Zpos[played_lin, torch.zeros_like(ply)]
        new_key = Zpos[played_lin, (ply + 1)]
        place_delta = torch.bitwise_xor(old_key, new_key)

        # Capture delta over all stones removed by this move
        cap_flat    = cap_mask_2d.view(M, N2)

        Z_emp_rows = self.ZposT[0].expand(M, -1)
        Z_opp_rows = self.ZposT[(opp + 1)]

        sel_opp = torch.where(cap_flat, Z_opp_rows, torch.zeros_like(Z_opp_rows))  # (M, N2)
        sel_emp = torch.where(cap_flat, Z_emp_rows, torch.zeros_like(Z_emp_rows))  # (M, N2)
        
        def xor_reduce_rowwise(x):
            M_, K_ = x.shape
            P = 1 << (K_ - 1).bit_length()
            if P != K_:
                x = torch.cat([x, x.new_zeros(M_, P - K_)], dim=1)
                K_ = P
            steps = K_.bit_length() - 1
            for _ in range(steps):
                half = K_ // 2
                x = torch.bitwise_xor(x[:, :half], x[:, half:half*2])
                if K_ & 1:
                    x = torch.bitwise_xor(x, x[:, -1:])
                K_ = x.size(1)
            return x.squeeze(1)
        
        cap_delta_opp = xor_reduce_rowwise(sel_opp)
        cap_delta_emp = xor_reduce_rowwise(sel_emp)
        cap_delta     = torch.bitwise_xor(cap_delta_opp, cap_delta_emp)

        self.current_hash[b_idx] ^= (place_delta ^ cap_delta)

    @timed_method
    def _place_stones(self, positions: PositionTensor) -> None:
        """Vectorized stone placement, capture handling, and incremental hash update (CSR)."""
        dev = self.device
        H = W = self.board_size
        N2 = H * W

        # Masked indexing works even with empty selections (no need to branch)
        mask_play = (positions[:, 0] >= 0) & (positions[:, 1] >= 0)   # (B,)
        b_idx = mask_play.nonzero(as_tuple=True)[0]                   # (M,)
        rows = positions[b_idx, 0].long()
        cols = positions[b_idx, 1].long()
        ply  = self.current_player[b_idx].long()                      # (M,)
        M    = b_idx.numel()

        # ---- Build capture mask for the played moves from CSR (branch-free) ----
        # CSR (flat row-space) from last legal computation
        cap_indptr  = self._last_info["cap_indptr"].reshape(-1).to(torch.int64)  # (B*N2+1,)
        cap_indices = self._last_info["cap_indices"].reshape(-1).to(torch.int64) # (L,)

        # Row id in the flat CSR space is r = b*N2 + i (i is linear move index)
        i_lin  = (rows * W + cols).to(torch.int64)              # (M,)
        r_flat = (b_idx.to(torch.int64) * N2 + i_lin)           # (M,)

        starts = cap_indptr[r_flat]                             # (M,)
        ends   = cap_indptr[r_flat + 1]                         # (M,)
        lens   = (ends - starts)                                # (M,) int64

        # Pad to Lmax and gather (works for lens==0 too)
        Lmax = int(lens.max())                           # small host read; ok
        offs = torch.arange(Lmax, device=dev, dtype=torch.int64)              # (Lmax,)
        idx  = starts[:, None] + offs[None, :]                               # (M, Lmax)
        mask = offs[None, :] < lens[:, None]                                  # (M, Lmax)

        # Gather captured linear indices, mask out padded cols
        jmat = torch.where(mask, cap_indices[idx], torch.zeros_like(idx))     # (M, Lmax)

        # Scatter into a (M, H, W) boolean mask without loops
        cap_mask_flat = torch.zeros(M * N2, dtype=torch.bool, device=dev)     # (M*N2,)
        base = (torch.arange(M, device=dev, dtype=torch.int64) * N2)[:, None] # (M,1)
        flat_pos = base + jmat                                                # (M, Lmax)
        flat_sel = flat_pos[mask]                                             # (K,) possibly 0
        cap_mask_flat[flat_sel] = True
        cap_mask_2d = cap_mask_flat.view(M, H, W)

        # ---- Incremental hash update BEFORE mutating the board -----------------
        if self.enable_super_ko:
            self._update_hash_incremental(b_idx, rows, cols, ply, cap_mask_2d)

        # ---- Apply the move on the board --------------------------------------
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

        flat = self.board.flatten(1)  # (B, H*W)
        

        move_idx = self.move_count.long()
        valid_mask = move_idx < max_moves
        
        b_idx = torch.arange(B, device=self.device)[valid_mask]
        mv_idx = move_idx[valid_mask]
        
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
    @timed_method
    def is_game_over(self) -> PassTensor:
        """Check if any games have ended (2 consecutive passes)."""
        return self.pass_count >= 2
    @timed_method
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
