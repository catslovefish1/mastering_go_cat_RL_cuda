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
        history_factor: int = 1,
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

            # -------------------- History tracking --------------------
        max_moves = H * W * self.history_factor

        # Keep board_history only for first up-to-16 boards (debug/printing)
        self._board_hist_track_B = min(B, 16)  # number of boards tracked for snapshots
        self.register_buffer(
            "board_history",
            torch.full(
                (self._board_hist_track_B, max_moves, H * W),
                -1, dtype=torch.int8, device=dev
            )
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
    
       # ======= Super-ko via CSR (no BxN2xN2 tensors) ============================
    @timed_method
    def _filter_super_ko_vectorized(self, legal_mask: BoardTensor, info: Dict) -> BoardTensor:
        """
        Super-ko pre-filter using CSR + per-group XOR deltas.
        No chunking and no (B,N2,M) intermediates: uses batched torch.searchsorted
        for history membership tests.
        """
        device = self.device
        B, H, W = legal_mask.shape
        N2 = H * W

        # ---------- Candidates ----------
        legal_flat: Tensor = legal_mask.view(B, N2)          # (B,N2)
        cand_mask: Tensor  = legal_flat.bool()               # (B,N2)

        # ---------- Zobrist placement delta (vectorized; no syncs) ----------
        Zpos: Tensor = self.Zpos                              # (N2,3) int32
        lin = torch.arange(N2, device=device).view(1, -1).expand(B, -1)  # (B,N2)
        player = self.current_player.long()                               # (B,)
        empty_col  = torch.zeros_like(player)[:, None]                    # (B,1)
        player_col = (player + 1)[:, None]                                # (B,1)
        place_old  = Zpos[lin, empty_col]                                 # (B,N2) int32
        place_new  = Zpos[lin, player_col]                                # (B,N2) int32
        place_delta = torch.bitwise_xor(place_old, place_new).contiguous()# (B,N2) int32

        # ---------- Per-board D[j] = Z_opp[j] ^ Z_emp[j] ----------
        opp = (1 - player).long()                                         # (B,)
        Z_emp_BN2 = self.ZposT[0].expand(B, -1)                           # (B,N2) int32
        Z_opp_BN2 = self.ZposT[(opp + 1)]                                 # (B,N2) int32
        D = torch.bitwise_xor(Z_opp_BN2, Z_emp_BN2).contiguous()          # (B,N2) int32

        # ---------- CSR from checker ----------
        members_all: Tensor = info["stone_global_index"].to(torch.int64)             # (K,)
        indptr_all:  Tensor = info["stone_global_pointer"].to(torch.int64)           # (R+1,)
        gptr:        Tensor = info["group_global_pointer_per_board"].to(torch.int64) # (B+1,)

        R = int(indptr_all.numel() - 1)
        K = int(members_all.numel())

        # ===== Case 1: no groups/stones (fast exit) =====
        if R == 0 or K == 0:
            info["group_xor_remove_delta"] = torch.zeros(0, dtype=torch.int32, device=device)
            cap_delta = torch.zeros(B, N2, dtype=torch.int32, device=device)
            new_hash = (self.current_hash[:, None] ^ place_delta ^ cap_delta).contiguous()  # (B,N2) int32

            # -------- History check (no chunks, batched searchsorted) --------
            M = self.hash_history.shape[1]
            HIST = self.hash_history                                                 # (B,M) int32
            L = self.move_count.clamp_max(M).to(torch.int64)                         # (B,)
            INT32_MIN = torch.iinfo(torch.int32).min

            # syntax-easy-win: build validity mask with a vectorized range
            valid_cols = (torch.arange(M, device=device).view(1, -1) < L.view(-1, 1))   # (B,M)
            HIST_masked = torch.where(valid_cols, HIST, torch.full_like(HIST, INT32_MIN))
            H_sorted, _ = torch.sort(HIST_masked, dim=1)                                # (B,M)
            start_valid = (M - L).view(-1, 1)                                           # (B,1)

            idx = torch.searchsorted(H_sorted, new_hash, right=True) - 1                # (B,N2)
            idx_clamped = idx.clamp_min(0)
            val = H_sorted.gather(1, idx_clamped)                                       # (B,N2)
            in_valid = (idx >= start_valid)
            is_repeat_flat = in_valid & (val == new_hash)
            is_repeat_flat &= cand_mask
            repeat_mask = is_repeat_flat.view(B, H, W)
            return legal_mask & ~repeat_mask

        # ===== Case 2: normal path with groups =====
        # group -> owning board via per-board group counts
        groups_per_board = (gptr[1:] - gptr[:-1]).to(torch.int64)                       # (B,)
        group_board_of_row = torch.repeat_interleave(
            torch.arange(B, device=device, dtype=torch.int64), groups_per_board
        )                                                                               # (R,)

        # per-stone owner board via its group's owner
        group_lengths = (indptr_all[1:] - indptr_all[:-1]).to(torch.int64)              # (R,)
        g_id_of_stone = torch.repeat_interleave(torch.arange(R, device=device, dtype=torch.int64),
                                                group_lengths)                          # (K,)
        stone_owner_board = group_board_of_row[g_id_of_stone]                           # (K,)

        # gather D[b, j] for each stone
        D_flat = D.view(-1)                                                             # (B*N2,)
        lin_idx = stone_owner_board * N2 + members_all                                  # (K,)
        D_per_stone = D_flat[lin_idx]                                                   # (K,) int32

        # XOR-reduce to one delta per group (ragged → no loops, small temp)
        Lmax = int(group_lengths.max().item())
        pad = torch.zeros(R, Lmax, dtype=torch.int32, device=device)                    # (R,Lmax)
        idxK = torch.arange(K, device=device, dtype=torch.int64)
        start_of_group_for_stone = indptr_all[:-1][g_id_of_stone]                       # (K,)
        pos_in_group = (idxK - start_of_group_for_stone).to(torch.int64)                # (K,)
        pad[g_id_of_stone, pos_in_group] = D_per_stone.to(torch.int32)

        width = Lmax
        acc = pad
        while width > 1:
            half = width // 2
            left  = acc[:, :half]
            right = acc[:, half:half*2]
            acc = torch.bitwise_xor(left, right)
            if (width & 1) == 1:
                acc = torch.bitwise_xor(acc, acc[:, -1:])
            width = acc.size(1)
        group_xor_remove_delta = acc.squeeze(1)                                        # (R,) int32

        # candidate capture: map local gid (≤4) → global row, gather XOR deltas
        cap_local = info["captured_group_local_index"].to(torch.int64)                 # (B,N2,4)
        valid = cap_local >= 0
        g_base = gptr[:-1].view(B, 1, 1)                                               # (B,1,1)
        g_global = (g_base + cap_local.clamp_min(0)).view(-1)                          # (B*N2*4,)

        cap_vals = torch.zeros_like(g_global, dtype=torch.int32, device=device)        # (B*N2*4,)
        mask_flat = valid.view(-1)
        cap_vals[mask_flat] = group_xor_remove_delta[g_global[mask_flat]]
        cap_vals = cap_vals.view(B, N2, 4)

        # syntax-easy-win: XOR 4 dirs directly (≤4 always)
        cap_delta = (cap_vals[..., 0] ^ cap_vals[..., 1] ^ cap_vals[..., 2] ^ cap_vals[..., 3])  # (B,N2)

        new_hash = (self.current_hash[:, None] ^ place_delta ^ cap_delta).contiguous()  # (B,N2) int32

        # -------- History check (no chunks, batched searchsorted) --------
        M = self.hash_history.shape[1]
        HIST = self.hash_history                                                       # (B,M) int32
        L = self.move_count.clamp_max(M).to(torch.int64)                               # (B,)
        INT32_MIN = torch.iinfo(torch.int32).min

        valid_cols = (torch.arange(M, device=device).view(1, -1) < L.view(-1, 1))      # (B,M)
        HIST_masked = torch.where(valid_cols, HIST, torch.full_like(HIST, INT32_MIN))
        H_sorted, _ = torch.sort(HIST_masked, dim=1)                                   # (B,M)
        start_valid = (M - L).view(-1, 1)                                              # (B,1)

        idx = torch.searchsorted(H_sorted, new_hash, right=True) - 1                   # (B,N2)
        idx_clamped = idx.clamp_min(0)
        val = H_sorted.gather(1, idx_clamped)                                          # (B,N2)
        in_valid = (idx >= start_valid)
        is_repeat_flat = in_valid & (val == new_hash)
        is_repeat_flat &= cand_mask
        repeat_mask = is_repeat_flat.view(B, H, W)

        # Store per-group remove deltas for the placement step (already int32)
        info["group_xor_remove_delta"] = group_xor_remove_delta                        # (R,)

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

    # ==================== MOVE EXECUTION (CSR) ================================ #
    @timed_method
    def _place_stones(self, positions: PositionTensor) -> None:
        """
        Vectorized stone placement and capture using CSR (no dense masks, no CPU syncs).
        Updates Zobrist hash incrementally using precomputed per-group deltas.
        """
        device = self.device
        H = W = self.board_size
        N2 = H * W

        # Which entries actually play (not pass & not forced-pass)
        mask_play = (positions[:, 0] >= 0) & (positions[:, 1] >= 0)   # (B,)
        b_idx = mask_play.nonzero(as_tuple=True)[0]                   # (M,)
        if b_idx.numel() == 0:
            return

        rows = positions[b_idx, 0].long()
        cols = positions[b_idx, 1].long()
        lin  = rows * W + cols                                        # (M,)
        ply  = self.current_player[b_idx].long()                      # (M,)

        # --- Pre-fetch CSR & capture lists from the last legal-move pass -------
        info = self._last_info
        cap_local = info["captured_group_local_index"]                 # (B,N2,4) int32
        gptr      = info["group_global_pointer_per_board"]             # (B+1,)   int32
        indptr    = info["stone_global_pointer"]                       # (R+1,)   int32
        members   = info["stone_global_index"]                         # (K,)     int32
        grp_xor   = info["group_xor_remove_delta"]                     # (R,)     int32

        # --- Zobrist placement delta for each actual move ----------------------
        Zpos: Tensor = self.Zpos                                       # (N2,3) int32
        old_key = Zpos[lin, torch.zeros_like(ply)]                     # (M,)   empty
        new_key = Zpos[lin, (ply + 1)]                                 # (M,)   player
        place_delta = torch.bitwise_xor(old_key, new_key)              # (M,)   int32

        # --- Gather up to 4 captured groups for every move in one shot ----------
        # local gids for the move's 4 neighbours
        g_local4 = cap_local[b_idx, lin]                               # (M,4) int32 (−1 if none)
        valid4   = (g_local4 >= 0)                                     # (M,4) bool

        # map to global group ids
        g_base_M = gptr[b_idx].to(g_local4.dtype).unsqueeze(1)         # (M,1)
        g_global4 = g_base_M + g_local4.clamp_min(0)                   # (M,4) int32

        # --- Hash update: XOR the ≤4 precomputed group removal deltas ----------
        # (invalid dirs contribute 0)
        cap_vals4 = torch.zeros_like(g_global4, dtype=torch.int32)
        if grp_xor.numel() > 0 and valid4.any():
            cap_vals4[valid4] = grp_xor[g_global4[valid4]]             # fill valid entries

        cap_delta = cap_vals4[:, 0] ^ cap_vals4[:, 1] ^ cap_vals4[:, 2] ^ cap_vals4[:, 3]  # (M,) int32
        # final hash update per move
        self.current_hash[b_idx] ^= (place_delta ^ cap_delta)

        # --- Build one big list of captured stones across all moves -------------
        # Flatten groups we actually capture this ply (M*≤4)
        g_global_flat = g_global4.view(-1)                              # (M*4,)
        valid_flat    = valid4.view(-1)                                 # (M*4,)
        g_list        = g_global_flat[valid_flat]                       # (L,)

        if g_list.numel() > 0:
            # Ranges inside members[] for each captured group
            starts = indptr[g_list]                                     # (L,)
            ends   = indptr[g_list + 1]                                 # (L,)
            lens   = (ends - starts)                                    # (L,)

            # Per-group parent move's board id (one move per board guarantee holds)
            groups_per_move = valid4.sum(1).to(torch.int64)             # (M,)
            board_of_group  = torch.repeat_interleave(b_idx.to(torch.int64), groups_per_move)  # (L,)

            # Expand to stone-level indices, fully vectorized (no .tolist(), no Python loop)
            S = int(lens.sum().item())
            if S > 0:
                # Which group each stone comes from
                g_of_stone = torch.repeat_interleave(torch.arange(g_list.numel(), device=device, dtype=torch.int64), lens)  # (S,)
                # Start offset per such group
                start_for_stone = starts[g_of_stone]                                   # (S,)
                # Position within group: 0..len-1 using prefix trick
                # build prefix starts and subtract
                prefix = torch.cumsum(torch.nn.functional.pad(lens, (1, 0)), 0)[:-1]   # (L,)  [0, l0, l0+l1, ...]
                pos_in_group = torch.arange(S, device=device, dtype=torch.int64) - prefix[g_of_stone]  # (S,)

                member_idx = start_for_stone + pos_in_group                            # (S,)
                captured_lin = members[member_idx].long()                               # (S,)

                # Per-stone owning board
                board_of_stone = torch.repeat_interleave(board_of_group, lens)         # (S,)

                # Scatter EMPTY onto boards in one shot
                flat_all = self.board.view(-1)                                         # (B*N2,)
                lin_board_cell = board_of_stone * N2 + captured_lin                    # (S,)
                flat_all[lin_board_cell] = torch.tensor(Stone.EMPTY, dtype=flat_all.dtype, device=device)

        # --- Finally place the stones (vectorized) --------------------------------
        self.board[b_idx, rows, cols] = ply.to(self.board.dtype)


    # ------------------------------------------------------------------ #
    # History                                                            #
    # ------------------------------------------------------------------ #
    @timed_method
    def _update_board_history(self) -> None:
        """
        Record current position:
        - board_history: only for first up-to-16 boards (debug)
        - hash_history : for all B boards (unchanged)
        """
        B, H, W = self.batch_size, self.board_size, self.board_size
        max_moves = self.board_history.shape[1] if self.board_history.numel() else 0

        # Flatten board once
        flat = self.board.flatten(1)  # (B, H*W)

        # Per-board move index
        move_idx = self.move_count.long()                     # (B,)
        # Valid if within max_moves (debug snapshots)
        valid_mask = (move_idx < max_moves) if max_moves > 0 else torch.zeros(B, dtype=torch.bool, device=self.device)

        # --- board_history: only write for boards < tracked_B ---
        T = self._board_hist_track_B
        if T > 0 and max_moves > 0:
            b_all   = torch.arange(B, device=self.device)[valid_mask]  # boards that have a valid snapshot slot
            mv_all  = move_idx[valid_mask]
            track_m = (b_all < T)                                      # only first T boards
            if track_m.any():
                b_trk  = b_all[track_m]
                mv_trk = mv_all[track_m]
                self.board_history[b_trk, mv_trk] = flat[b_trk]

        # --- hash_history: unchanged; write for all valid boards ---
        if self.enable_super_ko:
            max_moves_hash = self.hash_history.shape[1]
            valid_mask_hash = move_idx < max_moves_hash
            if valid_mask_hash.any():
                b_idx  = torch.arange(B, device=self.device)[valid_mask_hash]
                mv_idx = move_idx[valid_mask_hash]
                self.hash_history[b_idx, mv_idx] = self.current_hash[valid_mask_hash]


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

        # Which games are already over (two consecutive passes)?
        finished = (self.pass_count >= 2)
        active   = ~finished
        
        # Record history BEFORE the move
        self._update_board_history()
        self.move_count += 1
        
        # For finished games, force positions to "pass" so downstream is a no-op.
        forced_pass = torch.full_like(positions, -1)
        safe_positions = torch.where(finished.unsqueeze(1), forced_pass, positions)

        # Detect pass on the effective positions (active or forced)
        is_pass = (safe_positions[:, 0] < 0) | (safe_positions[:, 1] < 0)

        # Update pass_count ONLY for active games; clamp to 2 for a stable terminal state.
        inc_or_reset = torch.where(is_pass, self.pass_count + 1, torch.zeros_like(self.pass_count))
        new_pass_count = torch.where(active, inc_or_reset, self.pass_count).clamp_max(2)
        self.pass_count = new_pass_count  # store

        # Apply placement/captures for active, non-pass moves only.
        # (If your _place_stones already no-ops on pass, this is enough.)
        self._place_stones(safe_positions)


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
