# -*- coding: utf-8 -*-
"""
GoLegalMoveChecker.py – board-plane edition (CSR capture; no dense BxN2xN2)
============================================================================

Board
-----
- board: (B, H, W) int8 with values: -1 empty, 0 black, 1 white
- Internally we work on a flattened grid: N2 = H * W

CSR nomenclature used below
--------------------------
stone_global_index               : (K,)    int32  # all stone cell-ids, concatenated group-major
stone_global_pointer             : (R+1,)  int32  # CSR indptr over all groups in the batch
group_global_pointer_per_board   : (B+1,)  int32  # per-board offset of groups (local→global bridge)
stone_local_index_from_cell      : (B,N2)  int32  # (b, cell) → local gid (−1 for empty)
stone_local_index_from_root      : (B,N2)  int32  # (b, UF root) → local gid (−1 if no stones at that root)
captured_group_local_index       : (B,N2,4)int32  # per candidate cell, up to 4 capturable neighbor groups (−1 where none)
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict

import torch

from utils.shared import (
    timed_method,
)


# =============================================================================
# Public API
# =============================================================================

class GoLegalMoveChecker:
    def __init__(self, board_size: int = 19, device: Optional[torch.device] = None):
        self._checker = VectorizedBoardChecker(board_size, device)

    def compute_batch_legal_and_info(
        self,
        board: torch.Tensor,
        current_player: torch.Tensor,
        return_info: bool = True,
    ):
        B, H, W = board.shape
        assert H == W == self._checker.board_size, "board size mismatch"
        legal, info = self._checker.compute_batch_legal_and_info(board, current_player)
        return (legal, info) if return_info else legal


# =============================================================================
# Batched implementation
# =============================================================================

class VectorizedBoardChecker:
    """
    Fully batched legal-move logic with capture detection.
    Works on (B,H,W) int8 boards, flattens to (B,N2) internally.
    """

    # Per-board (flatten) structures (static for a given board size)
    index_flatten: torch.Tensor          # (N2,)
    neigh_index_flatten: torch.Tensor    # (N2,4) int64
    neigh_valid_flatten: torch.Tensor    # (N2,4) bool

    # Per-call (runtime) data
    board_flatten: torch.Tensor          # (B,N2), set each call

    def __init__(self, board_size: int, device: Optional[torch.device]):
        self.board_size = board_size
        self.N2 = board_size * board_size
        self.device = device

        # UF workspace (lazy, depends on B)
        self._uf_nbr_parent: Optional[torch.Tensor] = None  # (B,N2,4) int32

        # CSR debug state / workspaces
        self._csr_debug_id = 0
        self._csr_capacity_K = 0
        self._csr_capacity_R = 0
        self._csr_sg = None      # stone_global_index
        self._csr_sp = None      # stone_global_pointer
        self._csr_slc = None     # stone_local_index_from_cell
        self._csr_slr = None     # stone_local_index_from_root
        self._csr_gptr = None    # group_global_pointer_per_board

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

        # (N2,4) neighbours via offsets: N,S,W,E
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

        # Non-negative neighbor indices (off-board → 0) for gather
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
        board: torch.Tensor,          # (B,H,W) int8 in {-1,0,1}
        current_player: torch.Tensor  # (B,)    uint8 in {0,1}
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, H, W = board.shape
        N2 = self.N2

        # Per-call runtime flatten (depends on B)
        self.board_flatten = board.reshape(B, N2)       # (B,N2)
        empty = (self.board_flatten == -1)              # (B,N2) bool

        # ---- Neighbour colors: compute ONCE per call and reuse ----
        neighbor_colors = self._get_neighbor_colors_batch()  # (B,N2,4)

        # Groups (roots) + liberties (reuse neighbor_colors inside)
        roots, root_libs = self._batch_init_union_find(neighbor_colors)  # (B,N2) each

        # === Build the CSR + LUTs (batch-wide) ============================
        csr = self._build_group_csr(roots)
        stone_global_index = csr["stone_global_index"]               # (K,)
        stone_global_pointer = csr["stone_global_pointer"]           # (R+1,)
        group_global_pointer_per_board = csr["group_global_pointer_per_board"]  # (B+1,)
        stone_local_index_from_cell = csr["stone_local_index_from_cell"]        # (B,N2)
        stone_local_index_from_root = csr["stone_local_index_from_root"]        # (B,N2)

        # === Neighbour tables ==============================================
        neigh_valid_flatten_b = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)
        neighbor_roots = self._get_neighbor_roots_batch(roots)  # (B,N2,4) int32

        curr = current_player.view(B, 1, 1)  # (B,1,1)
        opp = 1 - curr

        # A) immediate liberties
        has_any_lib = (
            (neighbor_colors == -1) & neigh_valid_flatten_b
        ).any(dim=2)  # (B,N2)

        # B) captures: adjacent opponent group with exactly 1 liberty
        neighbor_roots_flat = neighbor_roots.reshape(B, -1).to(torch.int64)  # (B,N2*4)
        neighbor_root_index = neighbor_roots_flat.clamp(min=0)               # (B,N2*4)
        neighbor_libs_flat = root_libs.gather(1, neighbor_root_index)        # (B,N2*4)
        neighbor_libs = neighbor_libs_flat.view(B, N2, 4)                    # (B,N2,4)

        opponent_mask = (neighbor_colors == opp) & neigh_valid_flatten_b     # (B,N2,4) bool
        can_capture_edge = opponent_mask & (neighbor_libs == 1)              # (B,N2,4)
        can_capture_any = can_capture_edge.any(dim=2)                        # (B,N2)

        # C) friendly safe attachment
        friendly_neighbor = (neighbor_colors == curr) & neigh_valid_flatten_b    # (B,N2,4)
        friendly_safe = (friendly_neighbor & (neighbor_libs > 1)).any(dim=2)     # (B,N2)

        # D) final legality
        legal_flat = empty & (has_any_lib | can_capture_any | friendly_safe)
        legal_mask = legal_flat.view(B, H, W)                                  # (B,H,W)

        # ----- Capture metadata (CSR-based; no dense mask) ----------------
        # Map neighbour *roots* → local group ids (or -1); keep only capturing dirs
        captured_group_local_index_all = stone_local_index_from_root.gather(
            1,
            neighbor_roots.clamp(min=0).reshape(B, -1).to(torch.int64),
        ).view(B, N2, 4)                                                      # (B,N2,4) int32

        captured_group_local_index = torch.where(
            can_capture_edge,
            captured_group_local_index_all,
            torch.full_like(captured_group_local_index_all, -1, dtype=torch.int32),
        )                                                                      # (B,N2,4) int32

        # info payload (no (B,N2,N2) tensors; use CSR + per-candidate gids)
        info: Dict[str, torch.Tensor] = {
            # core group topology
            "roots": roots,                          # (B,N2) int32
            "root_libs": root_libs,                  # (B,N2) int32

            # legality helpers
            "can_capture_any": can_capture_any,      # (B,N2) bool
            "captured_group_local_index": captured_group_local_index.long(),  # (B,N2,4) int32

            # CSR (global, batch-wide)
            "stone_global_index": stone_global_index.long(),              # (K,)   int32
            "stone_global_pointer": stone_global_pointer.long(),          # (R+1,) int32
            "group_global_pointer_per_board": group_global_pointer_per_board.long(),  # (B+1,) int32

            # LUTs (fast ID→ID maps)
            "stone_local_index_from_cell": stone_local_index_from_cell.long(),   # (B,N2) int32
            "stone_local_index_from_root": stone_local_index_from_root.long(),   # (B,N2) int32
        }

        return legal_mask, info

    # ------------------------------------------------------------------
    # UF + liberties with reusable (B,N2,4) workspace
    # ------------------------------------------------------------------
    @timed_method
    def _batch_init_union_find(
        self,
        neighbor_colors: torch.Tensor,   # (B,N2,4) int8 – precomputed neighbour colors
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        roots     : (B,N2) int32  union-find representative per point
        root_libs : (B,N2) int32  liberty count per root id (index by root id)
        """
        board_flatten = self.board_flatten
        B, N2 = board_flatten.shape
        dev = self.device or board_flatten.device

        # ---- DEBUG: measure UF/lib step on MPS --------------------------------
        is_mps = (dev is not None and dev.type == "mps" and hasattr(torch, "mps"))
        if is_mps:
            drv_before_uf = _mps_driver_mb(dev)
            print(f"[MPS][UF] before={drv_before_uf:.1f} MB")

        # ---- Neighbour validity mask (view only) -------------------------------
        neigh_valid_flatten_b = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)

        # ---- Same-color adjacency (ignore empties; respect edges) --------------
        same_color_neighbor = (
            (neighbor_colors == board_flatten.unsqueeze(2))
            & (board_flatten.unsqueeze(2) != -1)
            & neigh_valid_flatten_b
        )  # (B,N2,4)

        # ---- Hook & compress (union-find) --------------------------------------
        parent0 = torch.arange(N2, dtype=torch.int32, device=dev)
        parent  = parent0.unsqueeze(0).repeat(B, 1).contiguous()  # (B,N2)

        parent = self._hook_and_compress(parent, same_color_neighbor)
        roots = parent                                                          # (B,N2)

        # ---- Count unique liberties per root -----------------------------------
        is_liberty_edge = (neighbor_colors == -1) & neigh_valid_flatten_b       # (B,N2,4)
        has_stone = (board_flatten != -1)                                       # (B,N2)
        stone_to_liberty_edge_mask = is_liberty_edge & has_stone.unsqueeze(2)   # (B,N2,4)

        # K = number of stone→empty edges across the batch
        batch_idx, cell_idx, dir_idx = torch.where(stone_to_liberty_edge_mask)  # each (K,)
        root_idx = roots[batch_idx, cell_idx]                                   # (K,) root id
        liberty_idx = self.neigh_index_flatten[cell_idx, dir_idx]              # (K,) liberty cell id

        # Deduplicate by (batch, root, liberty_point) then count uniques per root
        root_key = batch_idx * N2 + root_idx       # (K,)
        liberty_key = batch_idx * N2 + liberty_idx # (K,)
        root_liberty_pairs = torch.stack((root_key, liberty_key), dim=1)  # (K,2)

        pair_sort_key = (
            root_liberty_pairs[:, 0].to(torch.int64) * (N2 * B)
            + root_liberty_pairs[:, 1].to(torch.int64)
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

        root_libs = liberty_counts_flat.view(B, N2)                             # (B,N2)

        # ---- DEBUG: print K and driver delta -----------------------------------
        if is_mps:
            num_edges = int(stone_to_liberty_edge_mask.sum().item())   # total stone→liberty edges
            num_unique_pairs = int(unique_pairs.size(0))               # unique (root, liberty) pairs
            drv_after_uf = _mps_driver_mb(dev)
            print(
                f"[MPS][UF]  after={drv_after_uf:.1f} MB "
                f"Δ={drv_after_uf - drv_before_uf:+.1f} MB "
                f"K_edges={num_edges} Kuniq={num_unique_pairs}"
            )

        return roots, root_libs

    # ------------------------------------------------------------------
    # UF workspace helper + batched pointer-jumping (int32)
    # ------------------------------------------------------------------
    @timed_method
    def _ensure_uf_workspace(self, B: int, N2: int, dev: torch.device) -> torch.Tensor:
        """
        Ensure we have a reusable (B,N2,4) int32 workspace for UF neighbor parents.
        We treat the buffer as a *capacity* buffer:
        - allocate once with at least (B,N2,4)
        - reuse it forever
        - if we ever need larger B or N2, grow it, but never shrink.
        """
        ws = self._uf_nbr_parent

        required_shape = (B, N2, 4)

        # If no buffer yet, or on a different device / dtype, create fresh
        if ws is None or ws.device != dev or ws.dtype != torch.int32:
            ws = torch.empty(required_shape, dtype=torch.int32, device=dev)
            self._uf_nbr_parent = ws
            return ws

        # If existing buffer is too small in B or N2, grow capacity
        cur_B, cur_N2, cur_4 = ws.shape
        if cur_B < B or cur_N2 < N2 or cur_4 != 4:
            new_B = max(cur_B, B)
            new_N2 = max(cur_N2, N2)
            ws = torch.empty((new_B, new_N2, 4), dtype=torch.int32, device=dev)
            self._uf_nbr_parent = ws
            return ws[:B, :N2]

        # Otherwise, reuse existing buffer; just slice to requested view
        return ws[:B, :N2]
        
    @timed_method
    def _hook_and_compress(
        self,
        parent: torch.Tensor,              # (B,N2) int32
        same_color_neighbor: torch.Tensor  # (B,N2,4) bool – adjacency for N,S,W,E
    ) -> torch.Tensor:
        """
        Batched union–find with pointer jumping.
        """
        B, N2 = parent.shape
        dev = parent.device  # use actual tensor device (cuda/mps/cpu)
    
        # Precomputed neighbor indices (flat), clamped off-board to 0
        neighbor_index = self.neigh_index_flatten.view(1, N2, 4).expand(B, -1, -1).clamp(min=0)
    
        # Reusable (B,N2,4) int32 workspace (capacity-based, sliced)
        neighbor_parent_ws = self._ensure_uf_workspace(B, N2, dev)
    
        max_rounds = N2.bit_length() +100
        rounds_used = 0
    
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
    
            # invalid neighbors → sentinel N2 (bigger than any valid parent index)
            neighbor_parent_ws.masked_fill_(~same_color_neighbor, N2)
    
            # best neighbor parent per point
            min_neighbor_parent = neighbor_parent_ws.min(dim=2).values  # (B, N2)
            hooked = torch.minimum(parent, min_neighbor_parent)         # (B, N2)
    
            # pointer jumping: compress by one step
            parent_next = torch.gather(hooked, 1, hooked.long())        # (B, N2)
            rounds_used = i + 1
    
            # every 4 rounds, check convergence (this syncs the device)
            if (i & 3) == 3 and torch.equal(parent_next, parent):
                parent = parent_next
                # print(f"[UF] converged in {rounds_used} rounds (B={B}, N2={N2})")
                break
    
            parent = parent_next
        # else:
        #     # only hit if loop never breaks
        #     # print(f"[UF] no convergence after {max_rounds} rounds (B={B}, N2={N2})")
    
        return parent


    # ------------------------------------------------------------------
    # Build CSR + LUTs (global across batch; safe for K=0)
    # ------------------------------------------------------------------
    @torch.no_grad()
    @timed_method
    def _build_group_csr(self, roots: torch.Tensor):
        dev = self.device 
        B, N2 = self.board_flatten.shape

        # ---- fixed capacity (worst case for this (B, N2)) -----------------
        maxK = B * N2           # worst-case stones
        maxR = B * N2           # worst-case groups (overkill but simple)

        # Grow CSR workspaces if needed (only on first use / if B, N2 changed)
        if self._csr_capacity_K < maxK:
            self._csr_sg = torch.empty(maxK, dtype=torch.int32, device=dev)
            self._csr_slc = torch.full((B, N2), -1, dtype=torch.int32, device=dev)
            self._csr_slr = torch.full((B, N2), -1, dtype=torch.int32, device=dev)
            self._csr_capacity_K = maxK

        if self._csr_capacity_R < maxR:
            self._csr_sp = torch.empty(maxR + 1, dtype=torch.int32, device=dev)
            self._csr_gptr = torch.empty(B + 1, dtype=torch.int32, device=dev)
            self._csr_capacity_R = maxR

        # --- debug bookkeeping ----------------------------------------
        self._csr_debug_id += 1
        csr_id = self._csr_debug_id
        is_mps = (dev.type == "mps" and hasattr(torch, "mps"))

        drv_before = _mps_driver_mb(dev) if is_mps else 0.0
        if is_mps:
            print(f"[MPS][csr:{csr_id}] before_csr={drv_before:.1f} MB")

        # 1) take stones (exclude empties)
        has_stone = (self.board_flatten != -1)                     # (B,N2)
        stone_batch_idx, stone_cell_idx = has_stone.nonzero(as_tuple=True)  # (K,), (K,)
        stone_root_idx = roots[stone_batch_idx, stone_cell_idx]    # (K,)

        # 2) global sort by (board, root) → stones contiguous per group
        group_sort_key = stone_batch_idx * (N2 + 1) + stone_root_idx
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
        stone_local_index_from_cell.fill_(-1)
        if K > 0:
            lin_cells = (b_sorted * N2 + j_sorted).to(torch.int64)    # (K,)
            stone_local_index_from_cell.view(-1).index_put_(
                (lin_cells,),
                local_group_id_for_stone,
                accumulate=False,
            )

        stone_local_index_from_root = self._csr_slr   # (B,N2)
        stone_local_index_from_root.fill_(-1)
        if R > 0:
            root_id_for_run = r_sorted[group_start_indices]                       # (R,)
            lin_roots = (group_board_idx * N2 + root_id_for_run).to(torch.int64)  # (R,)
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
    def _get_neighbor_colors_batch(self) -> torch.Tensor:
        """Return neighbor colors pulled from self.board_flatten without per-call index clamps."""
        B, N2 = self.board_flatten.shape

        # Views only (no alloc): expand precomputed indices & validity
        idx = self.neigh_index_nonneg_flatten.view(1, N2, 4).expand(B, -1, -1)  # (B,N2,4) long (view)
        valid = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)       # (B,N2,4) bool (view)

        # Gather from a broadcasted view of the board (no data copy)
        board3 = self.board_flatten.unsqueeze(2).expand(-1, -1, 4)              # (B,N2,4) view
        out = torch.gather(board3, dim=1, index=idx).to(torch.int8)             # (B,N2,4) int8

        # Mark off-board neighbors distinctly
        out.masked_fill_(~valid, -2)
        return out
        
    @timed_method
    def _get_neighbor_roots_batch(self, roots: torch.Tensor) -> torch.Tensor:
        """Return neighbor union-find roots using precomputed non-negative indices; off-board = -1."""
        B, N2 = roots.shape

        # Views only (no alloc)
        idx = self.neigh_index_nonneg_flatten.view(1, N2, 4).expand(B, -1, -1)   # (B,N2,4) long (view)
        valid = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)        # (B,N2,4) bool (view)

        roots3 = roots.unsqueeze(2).expand(-1, -1, 4)                             # (B,N2,4) view
        gathered = torch.gather(roots3, dim=1, index=idx)                         # (B,N2,4)
        gathered.masked_fill_(~valid, -1)
        return gathered


