# board_tensor.py – GPU-optimized Go engine (delta-hash super-ko, no board clones)

from __future__ import annotations
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
# Constants / aliases
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Stone:
    BLACK: int = 0
    WHITE: int = 1
    EMPTY: int = -1


BoardTensor = Tensor     # (B, H, W)
PositionTensor = Tensor  # (B, 2)
PassTensor = Tensor      # (B,)


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

        self.batch_size = batch_size
        self.board_size = board_size
        self.history_factor = history_factor
        self.device = device or select_device()
        self.enable_timing = enable_timing
        self.enable_super_ko = enable_super_ko
        self.debug_place_trace = debug_place_trace  # kept for API symmetry, not used yet

        # Timing infra (exclusive / inclusive handled by timed_method)
        if enable_timing:
            self.timings = defaultdict(list)
            self.call_counts = defaultdict(int)
        else:
            self.timings = {}
            self.call_counts = {}

        # Core legal move checker (C++/PyTorch backend)
        self.legal_checker = GoLegalMoveChecker(
            board_size=board_size,
            device=self.device,
        )

        # Initialize Zobrist tables + mutable state
        self._init_zobrist_tables()
        self._init_state()

        # Cache for legal moves and capture info
        self._last_legal_mask: Optional[Tensor] = None
        self._last_info: Optional[Dict] = None

    # ------------------------------------------------------------------ #
    # Zobrist                                                            #
    # ------------------------------------------------------------------ #
    def _init_zobrist_tables(self) -> None:
        """Initialize Zobrist hash tables + linear helpers for super-ko."""
        if not self.enable_super_ko:
            return

        B = self.batch_size
        H = W = self.board_size
        self.N2 = H * W  # single-board area

        dev = self.device

        # Capture staging + workspace, allocated once and reused
        self._cap_vals = torch.zeros(
            (B, self.N2, 4), dtype=torch.int32, device=dev
        )
        self._cap_vals.fill_(0)  # touch pages

        # Reuse across calls: 4 × (B, N2) int32 workspaces
        self._ws_int32 = torch.zeros(
            (4, B, self.N2), dtype=torch.int32, device=dev
        )
        self._ws_int32.fill_(0)

        # Group XOR workspace for super-ko (capacity = B * N2)
        self._group_xor = torch.zeros(B * self.N2, dtype=torch.int32, device=dev)

        # Random Zobrist values for each position and state: empty(0), black(1), white(2)
        torch.manual_seed(42)  # reproducible profiling / debugging
        self.zobrist_table = torch.randint(
            0,
            2**31,
            (H, W, 3),
            dtype=torch.int32,
            device=dev,
        )

        # Flattened Zobrist: (N2, 3) and its transpose (3, N2)
        self.Zpos = self.zobrist_table.view(self.N2, 3).contiguous()
        self.ZposT = self.Zpos.transpose(0, 1).contiguous()  # (3, N2)

    # ------------------------------------------------------------------ #
    # Mutable state                                                      #
    # ------------------------------------------------------------------ #
    def _init_state(self) -> None:
        """Allocate board state, history, and per-game hash tracking."""
        B = self.batch_size
        H = W = self.board_size
        dev = self.device

        # Main board state
        self.register_buffer(
            "board",
            torch.full((B, H, W), Stone.EMPTY, dtype=torch.int8, device=dev),
        )

        # Current player (0=black, 1=white)
        self.register_buffer("current_player", torch.zeros(B, dtype=torch.int8, device=dev))

        # Pass counter (game ends at 2)
        self.register_buffer("pass_count", torch.zeros(B, dtype=torch.int8, device=dev))

        # -------------------- History tracking --------------------
        max_moves = H * W * self.history_factor

        # Keep board_history only for first up to 16 boards (debug/printing)
        self._board_hist_track_B = 16
        self.register_buffer(
            "board_history",
            torch.full(
                (self._board_hist_track_B, max_moves, H * W),
                -1,
                dtype=torch.int8,
                device=dev,
            ),
        )

        self.register_buffer("move_count", torch.zeros(B, dtype=torch.int16, device=dev))

        # Super-ko tracking with Zobrist hashing
        if self.enable_super_ko:
            # Current hash for each game
            self.register_buffer("current_hash", torch.zeros(B, dtype=torch.int32, device=dev))
            # History of hashes for super-ko detection
            self.register_buffer(
                "hash_history",
                torch.zeros((B, max_moves), dtype=torch.int32, device=dev),
            )

    # ==================== CORE UTILITIES ==================================== #

    def switch_player(self) -> None:
        """Switch current player and invalidate cached legal moves."""
        self.current_player = self.current_player ^ 1
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Clear cached legal moves and capture info."""
        self._last_legal_mask = None
        self._last_info = None

    # ==================== LEGAL MOVES ======================================= #

    @timed_method
    def legal_moves(self) -> BoardTensor:
        """Compute / return cached legal-move mask (with optional super-ko)."""
        if self._last_legal_mask is None:
            legal_mask, legal_info = self._compute_legal_core()

            if self.enable_super_ko:
                legal_mask = self._filter_super_ko_vectorized(legal_mask, legal_info)

            self._last_legal_mask = legal_mask
            self._last_info = legal_info

        return self._last_legal_mask

    @timed_method
    def _compute_legal_core(self):
        """Thin timed wrapper around GoLegalMoveChecker.compute_batch_legal_and_info."""
        return self.legal_checker.compute_batch_legal_and_info(
            board=self.board,
            current_player=self.current_player,
            return_info=True,
        )

    # ======= Super-ko via CSR (no BxN2xN2 tensors) ========================== #

    @timed_method
    @torch.no_grad()
    def _filter_super_ko_vectorized(
        self, legal_mask: BoardTensor, info: Dict
    ) -> BoardTensor:
        """
        Filter legal moves using super-ko:
        - Uses CSR group representation from `info`
        - Uses Zobrist hashing on batch (B) × board (N2) states
        """
        device = self.device
        B = legal_mask.size(0)
        N2 = self.N2

        # Fixed workspaces (allocated once in _init_zobrist_tables)
        hash_workspace = self._ws_int32
        placement_delta = hash_workspace[0].view(B, N2)  # (B, N2) int32
        capture_delta = hash_workspace[1].view(B, N2)    # (B, N2) int32
        candidate_hashes = hash_workspace[2].view(B, N2) # (B, N2) int32

        cap_vals = self._cap_vals                  # (B, N2, 4) int32
        group_xor_workspace = self._group_xor      # (B * N2,) int32 – slice [:R]

        # ------------------------------------------------------------------ #
        # 1) Per-move placement delta (Zobrist)                              #
        # ------------------------------------------------------------------ #
        current_player = self.current_player.to(torch.long)  # (B,)

        z_empty = self.ZposT[0]  # (N2,)
        z_black = self.ZposT[1]  # (N2,)
        z_white = self.ZposT[2]  # (N2,)

        placement_delta.zero_()
        placement_delta.add_(z_empty.view(1, N2))

        black_mask = (current_player == 0)
        white_mask = ~black_mask
        if black_mask.any():
            placement_delta[black_mask] ^= z_black.view(1, N2)
        if white_mask.any():
            placement_delta[white_mask] ^= z_white.view(1, N2)

        # ------------------------------------------------------------------ #
        # 2) CSR pieces                                                      #
        # ------------------------------------------------------------------ #
        stone_global_index = info["stone_global_index"]              # (K,) int32
        stone_global_pointer = info["stone_global_pointer"]          # (R+1,) int32
        group_global_pointer_per_board = info["group_global_pointer_per_board"]  # (B+1,) int32
        captured_group_local_index = info["captured_group_local_index"]          # (B, N2, 4) int32

        # scalar sizes (keep as Python ints)
        R = int(stone_global_pointer.numel() - 1)
        K = int(stone_global_index.numel())
        # max_group_size = int((stone_global_pointer[1:] - stone_global_pointer[:-1]).max().item()) if R > 0 else 0

        # Early exit: no groups / no stones → only placement delta
        if R == 0 or K == 0:
            info["group_xor_remove_delta"] = group_xor_workspace[:0]

            capture_delta.zero_()
            candidate_hashes.copy_(self.current_hash.view(B, 1))
            candidate_hashes ^= placement_delta

            repeat_mask = self._repeat_mask_from_history(candidate_hashes, legal_mask)
            return legal_mask & ~repeat_mask

        # ------------------------------------------------------------------ #
        # 3) Per-stone toggle for captures (stone -> board)                  #
        # ------------------------------------------------------------------ #
        z_empty_flat = self.ZposT[0]      # (N2,)
        z_by_color = self.ZposT[1:3]      # (2, N2)

        groups_per_board = (group_global_pointer_per_board[1:] - group_global_pointer_per_board[:-1])  # (B,) int32
        group_board_index = torch.repeat_interleave(
            torch.arange(B, device=device, dtype=torch.long),
            groups_per_board.to(torch.long),
        )  # (R,)

        stones_per_group = (stone_global_pointer[1:] - stone_global_pointer[:-1])  # (R,) int32

        group_id_for_stone = torch.repeat_interleave(
            torch.arange(R, device=device, dtype=torch.long),
            stones_per_group.to(torch.long),
        )  # (K,)

        stone_board_index = group_board_index[group_id_for_stone]            # (K,)
        opp_player = (1 - current_player)                                    # (B,)
        opp_color_for_stone = opp_player[stone_board_index]                  # (K,)

        # stone_global_index is int32; convert once for indexing
        stone_global_index_long = stone_global_index.to(torch.long)

        z_opp = z_by_color[opp_color_for_stone, stone_global_index_long]     # (K,)
        z_emp = z_empty_flat[stone_global_index_long]                        # (K,)
        per_stone_delta = (z_opp ^ z_emp).to(torch.int32)                   # (K,)

        # ------------------------------------------------------------------ #
        # 4) Bounded-memory group XOR (tiled)                                #
        # ------------------------------------------------------------------ #
        group_stone_start = stone_global_pointer[:-1].to(torch.long)   # (R,)
        stones_per_group_long = stones_per_group.to(torch.long)        # (R,)

        group_xor = group_xor_workspace[:R]
        group_xor.zero_()

        R_TILE = 2**12   # 4096 groups per tile
        W_TILE = 2**4    # 16 elements per reduction chunk
        tile_offsets = torch.arange(W_TILE, device=device, dtype=torch.long)

        if int(stones_per_group_long.max().item()) > 0:
            max_idx = per_stone_delta.numel() - 1
            for r0 in range(0, R, R_TILE):
                r1 = min(r0 + R_TILE, R)
                start_segment = group_stone_start[r0:r1]   # (Rt,)
                count_segment = stones_per_group_long[r0:r1]  # (Rt,)
                if int(count_segment.max().item()) == 0:
                    continue

                max_len = int(count_segment.max().item())
                for p in range(0, max_len, W_TILE):
                    rel = p + tile_offsets                         # (W_TILE,)
                    valid = rel[None, :] < count_segment[:, None]  # (Rt, W_TILE) bool
                    idx = start_segment[:, None] + rel[None, :]    # (Rt, W_TILE)
                    idx = idx.clamp_max(max_idx)
                    vals = per_stone_delta[idx]
                    vals = torch.where(valid, vals, vals.new_zeros(()))

                    acc = vals
                    width = acc.size(1)
                    while width > 1:
                        half = width // 2
                        acc = acc[:, :half] ^ acc[:, half:half * 2]
                        if width & 1:
                            acc = acc ^ acc[:, -1:]
                        width = acc.size(1)

                    group_xor[r0:r1] ^= acc.squeeze(1)

        # ------------------------------------------------------------------ #
        # 5) Candidate capture (B,N2,4) staging – reuse _cap_vals            #
        # ------------------------------------------------------------------ #
        has_capture = captured_group_local_index >= 0          # (B, N2, 4) bool

        # Per-board group offsets (B,1,1)
        group_offset_per_board = group_global_pointer_per_board[:-1].view(B, 1, 1).to(torch.long)

        # 1) Global group ids in-place: offset + local ids
        cap_vals.copy_(captured_group_local_index.clamp_min(0))  # (B, N2, 4)
        cap_vals.add_(group_offset_per_board)                     # global group ids

        # 2) Map global group ids -> per-group XOR deltas
        if group_xor.numel() and has_capture.any():
            flat_group_ids = cap_vals[has_capture].to(torch.long)  # (K',)
            cap_vals.zero_()
            cap_vals[has_capture] = group_xor[flat_group_ids]
        else:
            cap_vals.zero_()

        # 3) Reduce over 4 neighbours -> single delta per candidate cell
        capture_delta.zero_()
        capture_delta.copy_(cap_vals[..., 0])
        capture_delta ^= cap_vals[..., 1]
        capture_delta ^= cap_vals[..., 2]
        capture_delta ^= cap_vals[..., 3]

        # ------------------------------------------------------------------ #
        # 6) Finalize hashes + repeat filter                                 #
        # ------------------------------------------------------------------ #
        candidate_hashes.copy_(self.current_hash.view(B, 1))
        candidate_hashes ^= placement_delta
        candidate_hashes ^= capture_delta

        info["group_xor_remove_delta"] = group_xor
        repeat_mask = self._repeat_mask_from_history(candidate_hashes, legal_mask)
        return legal_mask & ~repeat_mask

    # --------- shared helper: history repeat mask (deduped) ------------------ #
    @torch.no_grad()
    @timed_method
    def _repeat_mask_from_history(
        self, candidate_hashes: Tensor, legal_mask: Tensor
    ) -> Tensor:
        """
        candidate_hashes: (B, N2) int32 candidate future hashes
        legal_mask:      (B, H, W) bool
        returns:         (B, H, W) bool mask where moves repeat a prior hash
        """
        B, H, W = legal_mask.shape
        M = self.hash_history.shape[1]

        hash_history = self.hash_history                 # (B, M) int32
        moves_played = self.move_count.clamp_max(M).to(torch.long)  # (B,)

        INT32_MIN = torch.iinfo(torch.int32).min
        indices = torch.arange(M, device=self.device).view(1, -1)   # (1, M)
        valid = indices < moves_played.view(-1, 1)                  # (B, M) bool

        masked_history = torch.where(
            valid,
            hash_history,
            torch.full_like(hash_history, INT32_MIN),
        )
        sorted_history, _ = torch.sort(masked_history, dim=1)       # (B, M)

        # Binary search; if not present, 'found' will differ from candidate_hash
        search_idx = (
            torch.searchsorted(sorted_history, candidate_hashes, right=True) - 1
        ).clamp_min(0)                                              # (B, N2)
        found = sorted_history.gather(1, search_idx)                # (B, N2) int32

        is_repeat_flat = (found == candidate_hashes)                # (B, N2) bool
        return is_repeat_flat.view(B, H, W)

    # ------------------------------------------------------------------ #
    # Placement & captures                                               #
    # ------------------------------------------------------------------ #
    @timed_method
    def _place_stones(self, positions: PositionTensor) -> None:
        """
        Vectorized stone placement & capture using CSR.
        Uses precomputed per-group XOR deltas from the last legal() call.
        """
        dev = self.device
        B = self.batch_size
        H = W = self.board_size
        N2 = H * W

        # Which entries actually play (skip passes / forced passes)
        play_mask = (positions[:, 0] >= 0) & (positions[:, 1] >= 0)  # (B,)
        active_idx = play_mask.nonzero(as_tuple=True)[0]             # (M,)
        if active_idx.numel() == 0:
            return

        rows = positions[active_idx, 0].to(torch.long)
        cols = positions[active_idx, 1].to(torch.long)
        lin_pos = rows * W + cols                                    # (M,)
        current_player = self.current_player[active_idx].to(torch.long)  # (M,)

        # ---- Cached CSR + per-group deltas from last legal() ----
        info = self._last_info
        captured_group_local_index = info["captured_group_local_index"]          # (B, N2, 4) int32
        group_global_pointer_per_board = info["group_global_pointer_per_board"]  # (B+1,) int32
        stone_global_pointer = info["stone_global_pointer"]                      # (R+1,) int32
        stone_global_index = info["stone_global_index"]                          # (K,)   int32
        group_xor_remove_delta = info["group_xor_remove_delta"].to(torch.int32)  # (R,)

        # ---- Incremental Zobrist hash update: placement + captures ----
        Zpos = self.Zpos                                             # (N2, 3)
        placement_delta = Zpos[lin_pos, 0] ^ Zpos[lin_pos, (current_player + 1)]  # (M,) int32

        # Up to 4 captured groups per move; map local->global group id
        local_group_ids_4 = captured_group_local_index[active_idx, lin_pos]      # (M, 4)
        valid_slots = (local_group_ids_4 >= 0)                                   # (M, 4) bool
        global_group_ids_4 = (
            group_global_pointer_per_board[active_idx].unsqueeze(1)
            + local_group_ids_4.clamp_min(0)
        )                                                                        # (M, 4) int32

        capture_group_xor_4 = torch.zeros_like(global_group_ids_4, dtype=torch.int32)  # (M, 4)
        if group_xor_remove_delta.numel() and valid_slots.any():
            capture_group_xor_4[valid_slots] = group_xor_remove_delta[
                global_group_ids_4[valid_slots].to(torch.long)
            ]
        capture_delta = (
            capture_group_xor_4[:, 0]
            ^ capture_group_xor_4[:, 1]
            ^ capture_group_xor_4[:, 2]
            ^ capture_group_xor_4[:, 3]
        )  # (M,) int32

        self.current_hash[active_idx] ^= (placement_delta ^ capture_delta)

        # ---- Apply captures: clear stones in captured groups ----
        flat_valid = valid_slots.view(-1)
        if flat_valid.any():
            captured_groups_flat = global_group_ids_4.view(-1)[flat_valid]       # (L,) int32

            group_start = stone_global_pointer[captured_groups_flat]             # (L,) int32
            group_end = stone_global_pointer[captured_groups_flat + 1]           # (L,) int32
            stones_per_group = (group_end - group_start)                         # (L,) int32

            # Which board each captured group belongs to
            groups_per_move = valid_slots.sum(1).to(torch.long)                  # (M,)
            board_for_group = torch.repeat_interleave(
                active_idx.to(torch.long), groups_per_move
            )                                                                    # (L,)

            total_captured = int(stones_per_group.sum().item())
            if total_captured > 0:
                # Reconstruct stone indices for all captured groups
                group_id_for_stone = torch.repeat_interleave(
                    torch.arange(captured_groups_flat.numel(), device=dev, dtype=torch.long),
                    stones_per_group.to(torch.long),
                )                                                                # (S,)
                start_for_stone = group_start[group_id_for_stone].to(torch.long) # (S,)
                prefix_lengths = torch.cumsum(
                    torch.nn.functional.pad(stones_per_group.to(torch.long), (1, 0)), 0
                )[:-1]                                                           # (L,)
                pos_in_group = (
                    torch.arange(total_captured, device=dev, dtype=torch.long)
                    - prefix_lengths[group_id_for_stone]
                )

                stone_index_in_csr = start_for_stone + pos_in_group              # (S,)
                stone_global_index_long = stone_global_index.to(torch.long)
                captured_lin = stone_global_index_long[stone_index_in_csr]       # (S,)
                board_for_stone = torch.repeat_interleave(
                    board_for_group, stones_per_group.to(torch.long)
                )                                                                # (S,)

                flat_board = self.board.view(-1)                                 # (B * N2,)
                lin_board_cell = board_for_stone * N2 + captured_lin             # (S,)
                flat_board[lin_board_cell] = torch.tensor(
                    Stone.EMPTY, dtype=flat_board.dtype, device=dev
                )

        # ---- Finally, place the new stones ----
        self.board[active_idx, rows, cols] = current_player.to(self.board.dtype)

    # ------------------------------------------------------------------ #
    # History                                                            #
    # ------------------------------------------------------------------ #
    @timed_method
    def _update_board_history(self) -> None:
        """
        Record current position:
        - board_history: only for first up-to-16 boards (debug)
        - hash_history : for all B boards
        """
        B = self.batch_size
        H = W = self.board_size

        max_moves = self.board_history.shape[1] if self.board_history.numel() else 0

        # Flatten board once
        board_flat = self.board.flatten(1)  # (B, H * W)

        # Per-board move index
        move_idx = self.move_count.to(torch.long)  # (B,)

        # Debug snapshots (board_history)
        if max_moves > 0:
            valid_snapshot = (move_idx < max_moves)
        else:
            valid_snapshot = torch.zeros(B, dtype=torch.bool, device=self.device)

        tracked_B = self._board_hist_track_B
        if tracked_B > 0 and max_moves > 0:
            all_boards = torch.arange(B, device=self.device)[valid_snapshot]
            all_moves = move_idx[valid_snapshot]
            track_mask = (all_boards < tracked_B)
            if track_mask.any():
                tracked_boards = all_boards[track_mask]
                tracked_moves = all_moves[track_mask]
                self.board_history[tracked_boards, tracked_moves] = board_flat[tracked_boards]

        # hash_history: write for all valid boards
        if self.enable_super_ko:
            max_moves_hash = self.hash_history.shape[1]
            valid_hash = (move_idx < max_moves_hash)
            if valid_hash.any():
                board_idx = torch.arange(B, device=self.device)[valid_hash]
                move_pos = move_idx[valid_hash]
                self.hash_history[board_idx, move_pos] = self.current_hash[valid_hash]

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
        # Normalize input to the board's device (prevents mixed-device indexing)
        positions = positions.to(self.device, non_blocking=True)

        if positions.dim() != 2 or positions.size(1) != 2:
            raise ValueError("positions must be (B, 2)")
        if positions.size(0) != self.batch_size:
            raise ValueError(
                f"batch size mismatch: expected {self.batch_size}, got {positions.size(0)}"
            )

        # Which games are already over (two consecutive passes)?
        finished = (self.pass_count >= 2)
        active = ~finished

        # Record history BEFORE the move
        self._update_board_history()
        self.move_count += 1

        # For finished games, force positions to "pass" so downstream is a no-op.
        forced_pass = torch.full_like(positions, -1)
        safe_positions = torch.where(finished.unsqueeze(1), forced_pass, positions)

        # Detect pass on the effective positions (active or forced)
        is_pass = (safe_positions[:, 0] < 0) | (safe_positions[:, 1] < 0)

        # Update pass_count ONLY for active games; clamp to 2 (terminal state stable).
        inc_or_reset = torch.where(is_pass, self.pass_count + 1, torch.zeros_like(self.pass_count))
        new_pass_count = torch.where(active, inc_or_reset, self.pass_count).clamp_max(2)
        self.pass_count = new_pass_count

        # Apply placement/captures for active, non-pass moves only.
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


