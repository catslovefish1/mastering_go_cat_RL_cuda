# -*- coding: utf-8 -*-
#engine/go_legal_checker.py
"""
batched Go rules engine (board-plane + CSR captures)
===========================================================================

Board
-----
- board: (B, H, W) int8 with values: Stone.EMPTY, Stone.BLACK, Stone.WHITE
- Internally we work on a flattened grid: N2 = H * W

CSR nomenclature
----------------
stone_global_index               : (K,)    int32  # all stone cell-ids, concatenated group-major
stone_global_pointer             : (R+1,)  int32  # CSR indptr over all groups in the batch
group_global_pointer_per_board   : (B+1,)  int32  # per-board offset of groups (local→global bridge)
stone_local_index_from_cell      : (B,N2)  int32  # (b, cell) → local group id (−1 for empty)
stone_local_index_from_root      : (B,N2)  int32  # (b, UF root) → local group id (−1 if no stones at that root)
captured_group_local_index       : (B,N2,4)int32  # per candidate cell, up to 4 capturable neighbour groups (−1 where none)

Public API
----------
class GoLegalChecker:
    - board_size: int
    - device: torch.device or None

    def compute_batch_legal_and_info(
        self,
        board: Tensor,          # (B,H,W) int8 in {Stone.EMPTY, Stone.BLACK, Stone.WHITE}
        move_color: Tensor,    # (B,)    int8, Stone.BLACK or Stone.WHITE
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]

Returns:
    legal_mask : (B,H,W) bool
    info       : dict with CSR + capture metadata (see above)
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict

from dataclasses import dataclass 

import torch
from torch import Tensor
from utils.shared import timed_method


@dataclass
class RulesLegalInfo:
    """
    Result of a rules-only legality / topology pass for a batch of boards.

    All tensors share the same lifetime: they are valid for exactly one
    board position (the one you passed into compute_batch_legal_and_info).
    """

    # main output (before ko filtering)
    legal_mask: Tensor                 # (B,H,W) bool

    # core group topology
    roots: Tensor                      # (B,N2) int32
    root_libs: Tensor                  # (B,N2) int32

    # legality helpers
    can_capture_any: Tensor            # (B,N2) bool
    captured_group_local_index: Tensor # (B,N2,4) int32

    # CSR (global, batch-wide)
    stone_global_index: Tensor                 # (K,)   int32
    stone_global_pointer: Tensor               # (R+1,) int32
    group_global_pointer_per_board: Tensor     # (B+1,) int32

    # LUTs (fast ID→ID maps)
    stone_local_index_from_cell: Tensor        # (B,N2) int32
    stone_local_index_from_root: Tensor        # (B,N2) int32




# ------------------------------------------------------------------
# Stone + sentinel constants (CURRENT ENCODING)
# ------------------------------------------------------------------


from .stones import Stone  


# neighbour_colors special value for off-board
OFF_BOARD_COLOR = -2

# sentinel indices for "nothing here"
NO_ROOT    = -1
NO_GROUP   = -1
NO_CAPTURE = -1



class GoLegalChecker:
    """
    Fully batched legal-move + capture logic for Go.

    Works on (B,H,W) int8 boards, flattens to (B,N2) internally.
    Keeps precomputed neighbour tables and UF/CSR workspaces for reuse.
    """

    # Per-board (flatten) structures (static for a given board size)
    index_flatten: Tensor               # (N2,)
    neigh_index_flatten: Tensor         # (N2,4) int64
    neigh_valid_flatten: Tensor         # (N2,4) bool
    neigh_index_nonneg_flatten: Tensor  # (N2,4) int64

    # Per-call (runtime) data
    board_flatten: Tensor               # (B,N2), set each call

    def __init__(self, board_size: int = 19, device: Optional[torch.device] = None):
        self.board_size = board_size
        self.N2 = board_size * board_size
        self.device = device

        # UF workspace (lazy, depends on B)
        self._uf_nbr_parent: Optional[Tensor] = None  # (B,N2,4) int32

        # CSR state / workspaces
        self._csr_debug_id = 0
        self._csr_capacity_K = 0
        self._csr_capacity_R = 0
        # self._csr_sg: Optional[Tensor] = None      # stone_global_index
        # self._csr_sp: Optional[Tensor] = None      # stone_global_pointer
        # self._csr_slc: Optional[Tensor] = None     # stone_local_index_from_cell
        # self._csr_slr: Optional[Tensor] = None     # stone_local_index_from_root
        # self._csr_gptr: Optional[Tensor] = None    # group_global_pointer_per_board

        self._init_flat_board_structure()

    # ------------------------------------------------------------------
    # Precomputed 4-neighbour tables in flat space
    # ------------------------------------------------------------------
    def _init_flat_board_structure(self) -> None:
        N = self.board_size
        N2 = self.N2
        dev = self.device

        # (N2,) flat indices of a single board
        self.index_flatten = torch.arange(N2, dtype=torch.int64, device=dev)

        # (N2,4) neighbour flat indices via offsets: N,S,W,E
        OFFSETS = torch.tensor([-N, N, -1, 1], dtype=torch.int64, device=dev)  # (4,)
        neighbor_indices = self.index_flatten[:, None] + OFFSETS               # (N2,4)

        # Edge handling
        valid = (neighbor_indices >= 0) & (neighbor_indices < N2)              # (N2,4)
        col = self.index_flatten % N
        valid[:, 2] &= col != 0            # W invalid at left edge
        valid[:, 3] &= col != N - 1        # E invalid at right edge

        self.neigh_index_flatten = torch.where(
            valid, neighbor_indices, torch.full_like(neighbor_indices, -1)
        )
        self.neigh_valid_flatten = valid

        # Non-negative neighbour indices (off-board → 0) for gather
        self.neigh_index_nonneg_flatten = torch.where(
            self.neigh_valid_flatten,
            self.neigh_index_flatten,
            torch.zeros_like(self.neigh_index_flatten),
        )  # (N2,4) int64

    # ------------------------------------------------------------------
    # Top-level: legal mask + capture metadata (CSR-based; no dense BxN2xN2)
    # ------------------------------------------------------------------
    @timed_method
    def compute_batch_legal_and_info(
        self,
        board: Tensor,          # (B,H,W) int8 in {Stone.EMPTY, Stone.BLACK, Stone.WHITE}
        move_color: Tensor,    # (B,)    int8, Stone.BLACK or Stone.WHITE
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        B, H, W = board.shape
        N2 = self.N2
        assert H == W == self.board_size, "board size mismatch"

        # Per-call runtime flatten (depends on B)
        self.board_flatten = board.reshape(B, N2)       # (B,N2)
        empty = (self.board_flatten == Stone.EMPTY)     # (B,N2) bool


        # ---- Neighbour colours: compute ONCE per call and reuse ----------
        neighbor_colors = self._get_neighbor_colors_batch()  # (B,N2,4)

        # Groups (roots) + liberties (reuse neighbour_colors inside)
        roots, root_libs = self._batch_init_union_find(neighbor_colors)  # (B,N2) each

        # === Build the CSR + LUTs (batch-wide) ============================
        csr = self._build_group_csr(roots)
        stone_global_index = csr["stone_global_index"]               # (K,)   int32
        stone_global_pointer = csr["stone_global_pointer"]           # (R+1,) int32
        group_global_pointer_per_board = csr["group_global_pointer_per_board"]  # (B+1,) int32
        stone_local_index_from_cell = csr["stone_local_index_from_cell"]        # (B,N2) int32
        stone_local_index_from_root = csr["stone_local_index_from_root"]        # (B,N2) int32

        # === Neighbour tables ==============================================
        neigh_valid_flatten_b = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)
        neighbor_roots = self._get_neighbor_roots_batch(roots)  # (B,N2,4) int32

        # colour-of-move
        curr = move_color.view(B, 1, 1)          # (B,1,1), Stone.*
        # explicit opposite colour, no "1 - curr" hack
        opp = torch.where(
            curr == Stone.BLACK,
            torch.full_like(curr, Stone.WHITE),
            torch.full_like(curr, Stone.BLACK),
        )

        # A) immediate liberties – at least one empty neighbour
        has_any_lib = (
            (neighbor_colors == Stone.EMPTY) & neigh_valid_flatten_b
        ).any(dim=2)  # (B,N2)

        # B) captures: adjacent opponent group with exactly 1 liberty
        neighbor_roots_flat = neighbor_roots.reshape(B, -1)              # (B,N2*4) int32
        neighbor_root_index = neighbor_roots_flat.clamp(min=0).to(torch.int64)  # (B,N2*4) long
        neighbor_libs_flat = root_libs.gather(1, neighbor_root_index)    # (B,N2*4) int32
        neighbor_libs = neighbor_libs_flat.view(B, N2, 4)                # (B,N2,4)

        opponent_mask = (neighbor_colors == opp) & neigh_valid_flatten_b # (B,N2,4) bool
        can_capture_edge = opponent_mask & (neighbor_libs == 1)          # (B,N2,4)
        can_capture_any = can_capture_edge.any(dim=2)                    # (B,N2)

        # C) friendly safe attachment: attach to friendly group with >1 liberty
        friendly_neighbor = (neighbor_colors == curr) & neigh_valid_flatten_b    # (B,N2,4)
        friendly_safe = (friendly_neighbor & (neighbor_libs > 1)).any(dim=2)     # (B,N2)

        # D) final legality in flat space
        legal_flat = empty & (has_any_lib | can_capture_any | friendly_safe)     # (B,N2)
        legal_mask = legal_flat.view(B, H, W)                                    # (B,H,W)

        # ----- Capture metadata (CSR-based; no dense mask) ----------------
        # Map neighbour *roots* → local group ids (or -1); keep only capturing dirs
        neighbor_roots_clamped = neighbor_roots.clamp(min=0).reshape(B, -1)      # (B,N2*4)
        captured_group_local_index_all = stone_local_index_from_root.gather(
            1, neighbor_roots_clamped.to(torch.int64)
        ).view(B, N2, 4)                                                         # (B,N2,4) int32

        captured_group_local_index = torch.where(
            can_capture_edge,
            captured_group_local_index_all,
            torch.full_like(captured_group_local_index_all, NO_CAPTURE, dtype=torch.int32),
        )                                                                        # (B,N2,4) int32

        # info payload (no (B,N2,N2) tensors; use CSR + per-candidate gids)
        info: Dict[str, Tensor] = {
            # core group topology
            "roots": roots,                          # (B,N2) int32
            "root_libs": root_libs,                  # (B,N2) int32

            # legality helpers
            "can_capture_any": can_capture_any,      # (B,N2) bool
            "captured_group_local_index": captured_group_local_index,  # (B,N2,4) int32

            # CSR (global, batch-wide)
            "stone_global_index": stone_global_index,                    # (K,)   int32
            "stone_global_pointer": stone_global_pointer,                # (R+1,) int32
            "group_global_pointer_per_board": group_global_pointer_per_board,  # (B+1,) int32

            # LUTs (fast ID→ID maps)
            "stone_local_index_from_cell": stone_local_index_from_cell,  # (B,N2) int32
            "stone_local_index_from_root": stone_local_index_from_root,  # (B,N2) int32
        }

        return legal_mask, info

    # ------------------------------------------------------------------
    # UF + liberties with reusable (B,N2,4) workspace
    # ------------------------------------------------------------------
    @timed_method
    def _batch_init_union_find(
        self,
        neighbor_colors: Tensor,   # (B,N2,4) int8 – precomputed neighbour colours
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        roots     : (B,N2) int32  union-find representative per point
        root_libs : (B,N2) int32  liberty count per root id (index by root id)
        """
        board_flatten = self.board_flatten
        B, N2 = board_flatten.shape
        dev = self.device or board_flatten.device

        # Neighbour validity mask (view only)
        neigh_valid_flatten_b = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)

        # Same-colour adjacency (ignore empties; respect edges)
        same_color_neighbor = (
            (neighbor_colors == board_flatten.unsqueeze(2))
            & (board_flatten.unsqueeze(2) != Stone.EMPTY)
            & neigh_valid_flatten_b
        )


        # Hook & compress (union-find)
        parent0 = torch.arange(N2, dtype=torch.int32, device=dev)
        parent  = parent0.unsqueeze(0).repeat(B, 1).contiguous()  # (B,N2)

        parent = self._hook_and_compress(parent, same_color_neighbor)
        roots = parent                                                          # (B,N2) int32

        # Count unique liberties per root
        is_liberty_edge = (neighbor_colors == Stone.EMPTY) & neigh_valid_flatten_b  # (B,N2,4)
        has_stone = (board_flatten != Stone.EMPTY)                                  # (B,N2)
        stone_to_liberty_edge_mask = is_liberty_edge & has_stone.unsqueeze(2)   # (B,N2,4)

        # K = number of stone→empty edges across the batch
        batch_idx, cell_idx, dir_idx = torch.where(stone_to_liberty_edge_mask)  # each (K,)
        root_idx = roots[batch_idx, cell_idx]                                   # (K,) int32
        liberty_idx = self.neigh_index_flatten[cell_idx, dir_idx]              # (K,) int64

        # Deduplicate by (batch, root, liberty_point) then count uniques per root
        root_key = batch_idx.to(torch.int64) * N2 + root_idx.to(torch.int64)       # (K,)
        liberty_key = batch_idx.to(torch.int64) * N2 + liberty_idx                 # (K,)
        root_liberty_pairs = torch.stack((root_key, liberty_key), dim=1)           # (K,2)

        pair_sort_key = (
            root_liberty_pairs[:, 0] * (N2 * B)
            + root_liberty_pairs[:, 1]
        )
        sort_perm = pair_sort_key.argsort()
        pairs_sorted = root_liberty_pairs[sort_perm]
        unique_pairs = torch.unique_consecutive(pairs_sorted, dim=0)            # (Kuniq,2)

        liberty_counts_flat = torch.zeros(B * N2, dtype=torch.int32, device=dev)   # (B*N2,)
        if unique_pairs.numel() > 0:
            liberty_counts_flat.scatter_add_(
                0,
                unique_pairs[:, 0],
                torch.ones(unique_pairs.size(0), dtype=torch.int32, device=dev),
            )

        root_libs = liberty_counts_flat.view(B, N2)                             # (B,N2) int32

        return roots, root_libs

    # ------------------------------------------------------------------
    # UF workspace helper + batched pointer-jumping (int32)
    # ------------------------------------------------------------------
    @timed_method
    def _ensure_uf_workspace(self, B: int, N2: int, dev: torch.device) -> Tensor:
        """
        One-time allocation of (B, N2, 4) workspace.
        Assumes B, N2, device stay constant for this checker.
        """
        if self._uf_nbr_parent is None:
            self._uf_nbr_parent = torch.empty((B, N2, 4), dtype=torch.int32, device=dev)
            return self._uf_nbr_parent
    
        # optional: very cheap sanity checks in debug phase
        # assert self._uf_nbr_parent.shape[:2] == (B, N2)
        # assert self._uf_nbr_parent.device == dev
    
        return self._uf_nbr_parent


    @timed_method
    @torch.no_grad()
    def _hook_and_compress(
        self,
        parent: Tensor,              # (B,N2) int32
        same_color_neighbor: Tensor  # (B,N2,4) bool – adjacency for N,S,W,E
    ) -> Tensor:
        """
        Batched union–find with pointer jumping.
        """
        B, N2 = parent.shape
        dev = parent.device  # use actual tensor device (cuda/mps/cpu)

        # Precomputed neighbour indices (flat), clamped off-board to 0
        neighbor_index = self.neigh_index_flatten.view(1, N2, 4).expand(B, -1, -1).clamp(min=0)

        # Reusable (B,N2,4) int32 workspace (capacity-based, sliced)
        neighbor_parent_ws = self._ensure_uf_workspace(B, N2, dev)

        # # Local workspace: (B,N2,4) int32
        # neighbor_parent_ws = torch.empty(
        #     (B, N2, 4),
        #     dtype=torch.int32,
        #     device=dev,
        # )


        max_rounds = N2.bit_length() + 100

        for i in range(max_rounds):
            # view only, no alloc
            parent_3d = parent.view(B, N2, 1).expand(-1, -1, 4)

            # neighbor_parent_ws[b, j, k] = parent[b, neighbor_index[b, j, k]]
            torch.take_along_dim(
                parent_3d,
                neighbor_index,
                dim=1,
                out=neighbor_parent_ws,
            )

            # invalid neighbours → sentinel N2 (bigger than any valid parent index)
            neighbor_parent_ws.masked_fill_(~same_color_neighbor, N2)

            # best neighbour parent per point
            min_neighbor_parent = neighbor_parent_ws.min(dim=2).values  # (B, N2)
            hooked = torch.minimum(parent, min_neighbor_parent)         # (B, N2)

            # pointer jumping: compress by one step
            parent_next = torch.gather(hooked, 1, hooked.to(torch.int64))  # (B, N2)
            parent_next = parent_next.to(torch.int32)

            # every 4 rounds, check convergence (this syncs the device)
            if (i & 3) == 3 and torch.equal(parent_next, parent):
                parent = parent_next
                break

            parent = parent_next

        return parent



    # ------------------------------------------------------------------
    # Territory (Chinese area scoring: empty regions -> black/white/neutral)
    # ------------------------------------------------------------------
    @timed_method
    def compute_territory(self, board: Tensor) -> Tensor:
        """
        Compute territory counts (empty points) for each colour using area scoring.

        Args
        ----
        board : (B,H,W) int8
        Values are Stone.EMPTY, Stone.BLACK, Stone.WHITE

        Returns
        -------
        territory : (B,2) int32
            territory[:,0] = black territory (empty points owned by black)
            territory[:,1] = white territory (empty points owned by white)
        """
        B, H, W = board.shape
        assert H == W == self.board_size, "board size mismatch"

        N2 = self.N2
        dev = board.device if self.device is None else self.device

        # Flatten board once for this call
        board_flatten = board.reshape(B, N2)               # (B,N2)
        self.board_flatten = board_flatten                 # re-use neighbour helpers

        # Empty points mask
        empties = (board_flatten == Stone.EMPTY)           # (B,N2) bool

        # If no empties at all, territory is zero
        if not empties.any():
            return torch.zeros((B, 2), dtype=torch.int32, device=dev)

        # Neighbour colours (uses self.board_flatten)
        neighbor_colors = self._get_neighbor_colors_batch()          # (B,N2,4) int8
        neigh_valid_b = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)

        # Union empty points into connected regions:
        # edge between j and neighbour k if both are empty
        same_region_edge = (
            empties.unsqueeze(2) &
            (neighbor_colors == Stone.EMPTY) &
            neigh_valid_b
        )                                                         # (B,N2,4) bool

        parent0 = torch.arange(N2, dtype=torch.int32, device=dev)
        parent  = parent0.unsqueeze(0).repeat(B, 1).contiguous()  # (B,N2)
        roots_all = self._hook_and_compress(parent, same_region_edge)  # (B,N2) int32

        # Only care about empty cells
        empty_batch_idx, empty_cell_idx = empties.nonzero(as_tuple=True)  # (E,), (E,)
        if empty_batch_idx.numel() == 0:
            return torch.zeros((B, 2), dtype=torch.int32, device=dev)

        region_root = roots_all[empty_batch_idx, empty_cell_idx].to(torch.int64)  # (E,)

        # Region key is (board_id, root_id) combined into a single index
        region_key_empty = empty_batch_idx.to(torch.int64) * N2 + region_root     # (E,)

        total_regions = B * N2

        # Region size = number of empty points in that region
        region_size_flat = torch.bincount(
            region_key_empty,
            minlength=total_regions,
        ).to(torch.int32)                                                        # (B*N2,)

        # For each empty point, look at neighbouring stones to determine
        # whether the region touches black and/or white.
        region_key_per_empty = region_key_empty                                  # (E,)
        neighbor_colors_empty = neighbor_colors[empty_batch_idx, empty_cell_idx, :]  # (E,4)

        # Expand region key per 4 neighbours
        region_key_tile = region_key_per_empty.repeat_interleave(4)              # (E*4,)
        colors_flat = neighbor_colors_empty.reshape(-1)                          # (E*4,)

        is_black = (colors_flat == Stone.BLACK)
        is_white = (colors_flat == Stone.WHITE)

        has_black_flat = torch.zeros(total_regions, dtype=torch.bool, device=dev)
        has_white_flat = torch.zeros(total_regions, dtype=torch.bool, device=dev)

        if is_black.any():
            black_keys = region_key_tile[is_black]
            has_black_flat[black_keys] = True

        if is_white.any():
            white_keys = region_key_tile[is_white]
            has_white_flat[white_keys] = True

        # Classification per region: black-only, white-only, or neutral (dame)
        black_only = has_black_flat & ~has_white_flat
        white_only = has_white_flat & ~has_black_flat

        # Territory per region = region_size if owned, else 0
        zero_like = torch.zeros_like(region_size_flat)
        black_territory_flat = torch.where(black_only, region_size_flat, zero_like)
        white_territory_flat = torch.where(white_only, region_size_flat, zero_like)

        # Sum per board
        black_territory = black_territory_flat.view(B, N2).sum(dim=1)  # (B,)
        white_territory = white_territory_flat.view(B, N2).sum(dim=1)  # (B,)

        return torch.stack([black_territory, white_territory], dim=1)  # (B,2)


    # ------------------------------------------------------------------
    # Build CSR + LUTs (global across batch; safe for K=0)
    # ------------------------------------------------------------------
    @timed_method
    def _build_group_csr(self, roots: Tensor):
        dev = self.device or self.board_flatten.device
        B, N2 = self.board_flatten.shape

        # ---- fixed capacity (worst case for this (B, N2)) -----------------
        maxK = B * N2           # worst-case stones
        maxR = B * N2           # worst-case groups (overkill but simple)

        # Grow CSR workspaces if needed (only on first use / if B, N2 changed)
        if self._csr_capacity_K < maxK:
            self._csr_sg = torch.empty(maxK, dtype=torch.int32, device=dev)
            self._csr_slc = torch.full((B, N2), NO_GROUP, dtype=torch.int32, device=dev)
            self._csr_slr = torch.full((B, N2), NO_GROUP, dtype=torch.int32, device=dev)
            self._csr_capacity_K = maxK

        if self._csr_capacity_R < maxR:
            self._csr_sp = torch.empty(maxR + 1, dtype=torch.int32, device=dev)
            self._csr_gptr = torch.empty(B + 1, dtype=torch.int32, device=dev)
            self._csr_capacity_R = maxR

        # 1) take stones (exclude empties)
        has_stone = (self.board_flatten != Stone.EMPTY)   # (B,N2)
        stone_batch_idx, stone_cell_idx = has_stone.nonzero(as_tuple=True)  # (K,), (K,)
        stone_root_idx = roots[stone_batch_idx, stone_cell_idx]    # (K,) int32

        # 2) global sort by (board, root) → stones contiguous per group
        group_sort_key = stone_batch_idx.to(torch.int64) * (N2 + 1) + stone_root_idx.to(torch.int64)
        sort_perm = group_sort_key.argsort()

        b_sorted = stone_batch_idx[sort_perm]                      # (K,)
        j_sorted = stone_cell_idx[sort_perm]                       # (K,)
        r_sorted = stone_root_idx[sort_perm]                       # (K,)
        K = b_sorted.numel()

        # 3) run boundaries for (board, root)
        same_group_as_prev = (
            (b_sorted == torch.roll(b_sorted, 1))
            & (r_sorted == torch.roll(r_sorted, 1))
        )
        is_group_start = (~same_group_as_prev) | (
            torch.arange(K, device=dev) == 0
        )                                                           # (K,) bool

        group_start_indices = torch.nonzero(is_group_start, as_tuple=True)[0]  # (R,)
        R = group_start_indices.numel()
        group_board_idx = b_sorted[group_start_indices]                            # (R,)
        group_index_range = torch.arange(R, device=dev, dtype=torch.int64)         # (R,)

        group_index_for_stone = is_group_start.to(torch.int64).cumsum(0) - 1       # (K,)
        stones_per_group = torch.bincount(
            group_index_for_stone.clamp_min(0),
            minlength=R,
        ).to(torch.int32)                                                          # (R,)

        # 4) board pointers (local→global bridge)
        groups_per_board = torch.bincount(
            group_board_idx.to(torch.int64),
            minlength=B,
        ).to(torch.int32)                                                          # (B,)

        group_global_pointer_per_board = self._csr_gptr[:B+1]
        group_global_pointer_per_board.zero_()
        if B > 0:
            group_global_pointer_per_board[1:] = groups_per_board.cumsum(0)
        board_first_global = group_global_pointer_per_board[:-1].to(torch.int64)  # (B,)

        local_group_id_for_run = (
            group_index_range
            - board_first_global.index_select(0, group_board_idx)
        ).to(torch.int32)                                                          # (R,)
        local_group_id_for_stone = local_group_id_for_run[group_index_for_stone]  # (K,)

        # 5) outputs (CSR arrays + LUTs) – via slices of workspaces
        stone_global_index = self._csr_sg[:K]
        if K > 0:
            stone_global_index.copy_(j_sorted.to(torch.int32))

        stone_global_pointer = self._csr_sp[:R+1]
        stone_global_pointer.zero_()
        if R > 0:
            stone_global_pointer[1:R+1] = stones_per_group.cumsum(0)

        stone_local_index_from_cell = self._csr_slc   # (B,N2)
        stone_local_index_from_cell.fill_(NO_GROUP)
        if K > 0:
            lin_cells = (b_sorted.to(torch.int64) * N2 + j_sorted.to(torch.int64))    # (K,)
            stone_local_index_from_cell.view(-1).index_put_(
                (lin_cells,),
                local_group_id_for_stone,
                accumulate=False,
            )

        stone_local_index_from_root = self._csr_slr   # (B,N2)
        stone_local_index_from_root.fill_(NO_GROUP)
        if R > 0:
            root_id_for_run = r_sorted[group_start_indices]                       # (R,)
            lin_roots = (
                group_board_idx.to(torch.int64) * N2
                + root_id_for_run.to(torch.int64)
            )                                                                     # (R,)
            stone_local_index_from_root.view(-1).index_put_(
                (lin_roots,),
                local_group_id_for_run,
                accumulate=False,
            )

        return {
            "stone_global_index": stone_global_index,                     # (K,)
            "stone_global_pointer": stone_global_pointer,                 # (R+1,)
            "group_global_pointer_per_board": group_global_pointer_per_board,  # (B+1,)
            "stone_local_index_from_cell": stone_local_index_from_cell,   # (B,N2)
            "stone_local_index_from_root": stone_local_index_from_root,   # (B,N2)
        }

    # ------------------------------------------------------------------
    # Neighbour helpers (batched, flat graph, 4 dirs)
    # ------------------------------------------------------------------
    @timed_method
    def _get_neighbor_colors_batch(self) -> Tensor:
        """Return neighbour colours pulled from self.board_flatten without per-call index clamps.

        Returns
        -------
        neighbour_colors : (B,N2,4) int8
            For each board, cell, direction (N,S,W,E):
            OFF_BOARD_COLOR  : off-board
            Stone.EMPTY      : empty
            Stone.BLACK      : black stone
            Stone.WHITE      : white stone
        """
        B, N2 = self.board_flatten.shape

        # Views only (no alloc): expand precomputed indices & validity
        idx = self.neigh_index_nonneg_flatten.view(1, N2, 4).expand(B, -1, -1)  # (B,N2,4) long (view)
        valid = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)       # (B,N2,4) bool (view)

        # Gather from a broadcasted view of the board (no data copy)
        board3 = self.board_flatten.unsqueeze(2).expand(-1, -1, 4)              # (B,N2,4) view
        out = torch.gather(board3, dim=1, index=idx).to(torch.int8)             # (B,N2,4) int8

        # Mark off-board neighbours distinctly
        out.masked_fill_(~valid, OFF_BOARD_COLOR)

        return out

    @timed_method
    def _get_neighbor_roots_batch(self, roots: Tensor) -> Tensor:
        """Return neighbour union-find roots using precomputed non-negative indices; off-board = off-board = NO_ROOT -1.

        Returns
        -------
        neighbour_roots : (B,N2,4) int32
        """
        B, N2 = roots.shape

        # Views only (no alloc)
        idx = self.neigh_index_nonneg_flatten.view(1, N2, 4).expand(B, -1, -1)   # (B,N2,4) long (view)
        valid = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)        # (B,N2,4) bool (view)

        roots3 = roots.unsqueeze(2).expand(-1, -1, 4)                             # (B,N2,4) view
        gathered = torch.gather(roots3, dim=1, index=idx)                         # (B,N2,4) int32
        gathered.masked_fill_(~valid, NO_ROOT)
        return gathered
#-----end of go_legal_checker.py------------
