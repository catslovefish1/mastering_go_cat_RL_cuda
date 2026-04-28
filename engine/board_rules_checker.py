# -*- coding: utf-8 -*-
# engine/board_rules_checker.py
"""
Batched Go rules engine with point-space helpers and CSR capture metadata.

Board representation: `(B, N2)` where `N2 = board_size ** 2`.

Two group-id layers coexist:
- board-local group ids: local ids inside a single board, stored in `(B, N2)` LUTs
- batch-global group ids: packed CSR ids across the whole batch, stored via pointers

Group-id conversion pipeline
----------------------------
Three identity layers appear in the code:
  point index (0..N2-1)            – intersection on one board
  local group id   (per board)     – assigned by _build_group_csr via union-find roots
  global group id  (across batch)  – local group id + group_offset_by_board[b]

The conversion path through the code:
  root point ──► local group id              (group_local_id_by_root_point)
  local group id + board offset  ──► global  (group_offset_by_board)
  global group id  ──► CSR stone range       (stone_pointer_by_group[g] .. stone_pointer_by_group[g+1])

CSR nomenclature
----------------
stone_point_ids               : (K,)    int32  # point ids, group-major over the whole batch
stone_pointer_by_group        : (R+1,)  int32  # CSR indptr over batch-global groups
group_offset_by_board         : (B+1,)  int32  # per-board offset: local group id → global group id
group_local_id_by_root_point  : (B,N2)  int32  # (board, root point) → local group id, -1 if absent
captured_group_local_ids      : (B,N2,4)int32  # per candidate point, up to 4 local capturable groups
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
from utils.shared import timed_method

from .stones import Stone


# neighbour_colors special value for off-board
OFF_BOARD_COLOR = -2

# sentinel indices for "nothing here"
NO_ROOT = -1
NO_GROUP = -1
NO_CAPTURE = -1


def _to_gather_index(index_tensor: Tensor) -> Tensor:
    """Replace -1 sentinels with 0 so the tensor can be used in gather/index ops."""
    return index_tensor.clamp_min(0).to(torch.int64)


@dataclass(slots=True)
class GroupCsr:
    """Batch-global CSR topology plus board-local root lookup."""
    stone_point_ids: Tensor                 # (K,) int32
    stone_pointer_by_group: Tensor          # (R+1,) int32
    group_offset_by_board: Tensor           # (B+1,) int32
    group_local_id_by_root_point: Tensor    # (B,N2) int32


@dataclass(slots=True)
class LegalInfo:
    """Capture metadata and CSR topology used by GameStateMachine."""
    csr: GroupCsr
    captured_group_local_ids: Tensor        # (B,N2,4) int32


class BoardRulesChecker:
    """
    Fully batched board-point legality + capture logic for Go.

    Accepts `(B, N2)` int8 boards and works in point space throughout the
    legality, union-find, and CSR pipelines.

    This class only answers "which board intersections are legal to place
    a stone on?" given the current board state. Higher-level action concepts
    (pass, stop, resign, terminal no-op) are handled by the caller.
    """

    # Per-board point-neighbor tables (static for a given board size)
    point_ids: Tensor                   # (N2,)
    neighbor_point_ids: Tensor          # (N2,4) int64
    neighbor_on_board: Tensor           # (N2,4) bool
    neighbor_point_ids_safe: Tensor     # (N2,4) int64

    # Per-call (runtime) data
    boards: Tensor                      # (B,N2), set each call

    def __init__(
        self,
        board_size: int = 19,
        device: Optional[torch.device] = None,
    ):
        self.board_size = board_size
        self.N2 = board_size * board_size
        self.device = device

        # UF workspace (lazy, depends on B)
        self._uf_neighbor_parent: Optional[Tensor] = None  # (B,N2,4) int32

        # CSR workspace buffers (lazy, sized by batch and board area)
        self._csr_capacity_BN2 = 0
        self._csr_capacity_R = 0

        self._init_point_neighbor_tables()

    # ------------------------------------------------------------------
    # Precomputed 4-neighbour tables in point space
    # ------------------------------------------------------------------
    def _init_point_neighbor_tables(self) -> None:
        N = self.board_size
        N2 = self.N2
        dev = self.device

        # (N2,) point ids of a single board
        self.point_ids = torch.arange(N2, dtype=torch.int64, device=dev)

        # (N2,4) neighbour point ids via offsets: N,S,W,E
        OFFSETS = torch.tensor([-N, N, -1, 1], dtype=torch.int64, device=dev)  # (4,)
        neighbor_indices = self.point_ids[:, None] + OFFSETS                   # (N2,4)

        # Edge handling
        valid = (neighbor_indices >= 0) & (neighbor_indices < N2)              # (N2,4)
        col = self.point_ids % N
        valid[:, 2] &= col != 0            # W invalid at left edge
        valid[:, 3] &= col != N - 1        # E invalid at right edge

        self.neighbor_point_ids = torch.where(
            valid, neighbor_indices, torch.full_like(neighbor_indices, -1)
        )
        self.neighbor_on_board = valid

        # Non-negative neighbour ids (off-board -> 0) for gather
        self.neighbor_point_ids_safe = torch.where(
            self.neighbor_on_board,
            self.neighbor_point_ids,
            torch.zeros_like(self.neighbor_point_ids),
        )  # (N2,4) int64

        # int32 identity used as initial parent array in union-find
        self.point_ids_i32 = torch.arange(N2, dtype=torch.int32, device=dev)

    # ------------------------------------------------------------------
    # Top-level: board-point legality + capture metadata (CSR-based)
    # ------------------------------------------------------------------
    @timed_method
    def compute_batch_legal_and_info(
        self,
        board: Tensor,          # (B,N2) int8 – canonical board storage
        to_play_color: Tensor,  # (B,) int8, Stone.BLACK or Stone.WHITE
    ) -> Tuple[Tensor, LegalInfo]:
        """
        Returns
        -------
        legal_points : (B, N2) bool
            Placement mask over board points (pass not included).
        legal_info : LegalInfo
            Capturable-group metadata plus batch-global CSR topology.
        """
        B, N2 = board.shape
        assert N2 == self.N2, "board size mismatch"

        self.boards = board
        empty_points = (board == Stone.EMPTY)     # (B,N2) bool

        # ---- 1) Neighbour colours + union-find roots/liberties -----------
        neighbor_colors = self._get_neighbor_colors_batch()  # (B,N2,4)
        roots, root_libs = self._batch_init_union_find(neighbor_colors)  # (B,N2) each

        # ---- 2) Build board-local LUTs + batch-global CSR ---------------
        group_csr = self._build_group_csr(roots)
        group_local_id_by_root_point = group_csr.group_local_id_by_root_point

        # ---- 3) Board-point legality (point space) -------------------
        neighbor_on_board_b = self.neighbor_on_board.view(1, N2, 4).expand(B, -1, -1)
        neighbor_root_ids = self._get_neighbor_roots_batch(roots)  # (B,N2,4) int32

        to_play_color_3d = to_play_color.view(B, 1, 1)     # (B,1,1), Stone.*
        opponent_color_3d = (3 - to_play_color_3d)          # BLACK(1)↔WHITE(2)

        # A) immediate liberties – at least one empty neighbour
        has_any_lib = (
            (neighbor_colors == Stone.EMPTY) & neighbor_on_board_b
        ).any(dim=2)  # (B,N2)

        # B) captures: adjacent opponent group with exactly 1 liberty
        neighbor_root_ids_packed = neighbor_root_ids.reshape(B, -1)            # (B,N2*4) int32
        neighbor_root_gather_index = _to_gather_index(neighbor_root_ids_packed)  # (B,N2*4) int64
        neighbor_libs_packed = root_libs.gather(1, neighbor_root_gather_index)  # (B,N2*4) int32
        neighbor_libs = neighbor_libs_packed.view(B, N2, 4)                # (B,N2,4)

        opponent_neighbor = (neighbor_colors == opponent_color_3d) & neighbor_on_board_b
        can_capture_edge = opponent_neighbor & (neighbor_libs == 1)      # (B,N2,4)
        can_capture_any = can_capture_edge.any(dim=2)                    # (B,N2)

        # C) friendly safe attachment: attach to friendly group with >1 liberty
        friendly_neighbor = (neighbor_colors == to_play_color_3d) & neighbor_on_board_b
        friendly_safe = (friendly_neighbor & (neighbor_libs > 1)).any(dim=2)     # (B,N2)

        legal_points = empty_points & (has_any_lib | can_capture_any | friendly_safe)

        # ---- 4) Capture metadata (board-local) for downstream CSR consumers -
        # Map neighbor root → board-local group id; only keep slots that actually
        # capture.  Downstream code (GameStateMachine) lifts these local ids into
        # batch-global ids via group_offset_by_board before dereferencing
        # the CSR stone arrays.
        captured_group_local_ids_all = group_local_id_by_root_point.gather(
            1, _to_gather_index(neighbor_root_ids.reshape(B, -1))
        ).view(B, N2, 4)                                                         # (B,N2,4) int32

        captured_group_local_ids = torch.where(
            can_capture_edge,
            captured_group_local_ids_all,
            torch.full_like(captured_group_local_ids_all, NO_CAPTURE, dtype=torch.int32),
        )                                                                        # (B,N2,4) int32

        legal_info = LegalInfo(
            csr=group_csr,
            captured_group_local_ids=captured_group_local_ids,
        )
        return legal_points, legal_info

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
        roots     : (B,N2) int32  union-find representative per board point
        root_libs : (B,N2) int32  liberty count per root id (index by root id)
        """
        boards = self.boards
        B, N2 = boards.shape
        dev = self.device or boards.device

        # Neighbour validity mask (view only)
        neighbor_on_board_b = self.neighbor_on_board.view(1, N2, 4).expand(B, -1, -1)

        # Same-colour adjacency (ignore empties; respect edges)
        same_color_edge = (
            (neighbor_colors == boards.unsqueeze(2))
            & (boards.unsqueeze(2) != Stone.EMPTY)
            & neighbor_on_board_b
        )

        # Hook & compress (union-find)
        parent = self.point_ids_i32.unsqueeze(0).repeat(B, 1).contiguous()  # (B,N2)

        parent = self._hook_and_compress(parent, same_color_edge)
        roots = parent                                                          # (B,N2) int32

        # Count unique liberties per root
        is_liberty_edge = (neighbor_colors == Stone.EMPTY) & neighbor_on_board_b  # (B,N2,4)
        has_stone = (boards != Stone.EMPTY)                                     # (B,N2)
        stone_to_liberty_edge_mask = is_liberty_edge & has_stone.unsqueeze(2)   # (B,N2,4)

        # K = number of stone→empty edges across the batch
        batch_idx, point_idx, dir_idx = torch.where(stone_to_liberty_edge_mask)  # each (K,)
        root_idx = roots[batch_idx, point_idx]                                   # (K,) int32
        liberty_idx = self.neighbor_point_ids[point_idx, dir_idx]                # (K,) int64

        # Deduplicate by (batch, root, liberty point) then count uniques per root
        root_flat_idx = batch_idx.to(torch.int64) * N2 + root_idx.to(torch.int64)     # (K,)
        liberty_flat_idx = batch_idx.to(torch.int64) * N2 + liberty_idx               # (K,)
        root_liberty_pairs = torch.stack((root_flat_idx, liberty_flat_idx), dim=1)     # (K,2)

        pair_sort_key = (
            root_liberty_pairs[:, 0] * (N2 * B)
            + root_liberty_pairs[:, 1]
        )
        sort_perm = pair_sort_key.argsort()
        pairs_sorted = root_liberty_pairs[sort_perm]
        unique_pairs = torch.unique_consecutive(pairs_sorted, dim=0)            # (Kuniq,2)

        liberty_counts_by_root = torch.zeros(B * N2, dtype=torch.int32, device=dev)   # (B*N2,)
        if unique_pairs.numel() > 0:
            liberty_counts_by_root.scatter_add_(
                0,
                unique_pairs[:, 0],
                torch.ones(unique_pairs.size(0), dtype=torch.int32, device=dev),
            )

        root_libs = liberty_counts_by_root.view(B, N2)                             # (B,N2) int32

        return roots, root_libs

    # ------------------------------------------------------------------
    # UF workspace helper + batched pointer-jumping (int32)
    # ------------------------------------------------------------------
    @timed_method
    def _ensure_uf_workspace(self, B: int, N2: int, dev: torch.device) -> Tensor:
        """
        Reusable allocation of (B, N2, 4) workspace.
        Assumes B, N2, device stay constant for this checker.
        """
        if self._uf_neighbor_parent is None:
            self._uf_neighbor_parent = torch.empty((B, N2, 4), dtype=torch.int32, device=dev)
            return self._uf_neighbor_parent

        return self._uf_neighbor_parent

    @timed_method
    @torch.no_grad()
    def _hook_and_compress(
        self,
        parent: Tensor,              # (B,N2) int32
        same_color_neighbor: Tensor  # (B,N2,4) bool – adjacency for N,S,W,E
    ) -> Tensor:
        """
        Batched union-find with pointer jumping.
        """
        B, N2 = parent.shape
        dev = parent.device  # use actual tensor device (cuda/mps/cpu)

        # Precomputed neighbour gather indices: off-board neighbours already map to 0
        neighbor_gather_index = self.neighbor_point_ids_safe.view(1, N2, 4).expand(B, -1, -1)

        # Reusable (B,N2,4) int32 workspace (capacity-based, sliced)
        neighbor_parent_buffer = self._ensure_uf_workspace(B, N2, dev)
        max_rounds = N2.bit_length() + 100

        for i in range(max_rounds):
            # view only, no alloc
            parent_3d = parent.view(B, N2, 1).expand(-1, -1, 4)

            # neighbor_parent_buffer[b, point_id, dir] = parent[b, neighbor_point_id]
            torch.take_along_dim(
                parent_3d,
                neighbor_gather_index,
                dim=1,
                out=neighbor_parent_buffer,
            )

            # invalid neighbours -> sentinel N2 (bigger than any valid parent index)
            neighbor_parent_buffer.masked_fill_(~same_color_neighbor, N2)

            # best neighbour parent per point
            min_neighbor_parent = neighbor_parent_buffer.min(dim=2).values  # (B, N2)
            hooked = torch.minimum(parent, min_neighbor_parent)              # (B, N2)

            # pointer jumping: compress by one step
            parent_next = torch.gather(hooked, 1, hooked.to(torch.int64))    # (B, N2)
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
        board : (B,N2) int8
            Values are Stone.EMPTY, Stone.BLACK, Stone.WHITE

        Returns
        -------
        territory : (B,2) int32
            territory[:,0] = black territory (empty points owned by black)
            territory[:,1] = white territory (empty points owned by white)
        """
        B, N2 = board.shape
        assert N2 == self.N2, "board size mismatch"
        dev = board.device if self.device is None else self.device

        self.boards = board                                # re-use neighbour helpers

        # Empty points mask
        empties = (board == Stone.EMPTY)                   # (B,N2) bool

        # Neighbour colours (uses self.boards)
        neighbor_colors = self._get_neighbor_colors_batch()          # (B,N2,4) int8
        neighbor_on_board_b = self.neighbor_on_board.view(1, N2, 4).expand(B, -1, -1)

        # Union empty points into connected regions:
        # edge between j and neighbour k if both are empty
        same_region_edge = (
            empties.unsqueeze(2)
            & (neighbor_colors == Stone.EMPTY)
            & neighbor_on_board_b
        )                                                         # (B,N2,4) bool

        parent = self.point_ids_i32.unsqueeze(0).repeat(B, 1).contiguous()   # (B,N2)
        roots = self._hook_and_compress(parent, same_region_edge)  # (B,N2) int32

        # Only care about empty points
        empty_batch_idx, empty_point_idx = empties.nonzero(as_tuple=True)  # (E,), (E,)
        if empty_batch_idx.numel() == 0:
            return torch.zeros((B, 2), dtype=torch.int32, device=dev)

        region_root = roots[empty_batch_idx, empty_point_idx].to(torch.int64)  # (E,)

        # Region key is (board_id, root_id) combined into a single index
        region_key_empty = empty_batch_idx.to(torch.int64) * N2 + region_root      # (E,)
        total_regions = B * N2

        # Region size = number of empty points in that region
        region_size_by_region = torch.bincount(
            region_key_empty,
            minlength=total_regions,
        ).to(torch.int32)                                                           # (B*N2,)

        # For each empty point, look at neighbouring stones to determine
        # whether the region touches black and/or white.
        neighbor_colors_empty = neighbor_colors[empty_batch_idx, empty_point_idx, :]  # (E,4)

        # Expand region key per 4 neighbours
        region_key_tile = region_key_empty.repeat_interleave(4)                      # (E*4,)
        neighbor_colors_packed = neighbor_colors_empty.reshape(-1)                    # (E*4,)

        is_black = neighbor_colors_packed == Stone.BLACK
        is_white = neighbor_colors_packed == Stone.WHITE

        has_black_by_region = torch.zeros(total_regions, dtype=torch.bool, device=dev)
        has_white_by_region = torch.zeros(total_regions, dtype=torch.bool, device=dev)

        has_black_by_region[region_key_tile[is_black]] = True
        has_white_by_region[region_key_tile[is_white]] = True

        # Classification per region: black-only, white-only, or neutral (dame)
        black_only = has_black_by_region & ~has_white_by_region
        white_only = has_white_by_region & ~has_black_by_region

        # Territory per region = region_size if owned, else 0
        zero_like = torch.zeros_like(region_size_by_region)
        black_territory_by_region = torch.where(black_only, region_size_by_region, zero_like)
        white_territory_by_region = torch.where(white_only, region_size_by_region, zero_like)

        # Sum per board
        black_territory = black_territory_by_region.view(B, N2).sum(dim=1)  # (B,)
        white_territory = white_territory_by_region.view(B, N2).sum(dim=1)  # (B,)

        return torch.stack([black_territory, white_territory], dim=1)        # (B,2)

    # ------------------------------------------------------------------
    # Build CSR + LUTs (global across batch; safe for K=0)
    # ------------------------------------------------------------------
    @timed_method
    def _build_group_csr(self, roots: Tensor) -> GroupCsr:
        """
        Build board-local root-group lookup plus batch-global group CSR.

        ``group_offset_by_board[b]`` converts a board-local group id into a
        batch-global group id: ``global = local + group_offset_by_board[b]``.
        ``stone_pointer_by_group[global]..stone_pointer_by_group[global+1]``
        gives the slice in ``stone_point_ids`` for that group.
        """
        dev = self.device or self.boards.device
        B, N2 = self.boards.shape

        # ---- fixed capacity (worst case for this (B, N2)) -----------------
        maxK = B * N2           # worst-case stones
        maxR = B * N2           # worst-case groups (overkill but simple)

        if self._csr_capacity_BN2 < maxK:
            self._stone_point_ids_buf = torch.empty(maxK, dtype=torch.int32, device=dev)
            self._group_local_id_buf = torch.full((B, N2), NO_GROUP, dtype=torch.int32, device=dev)
            self._csr_capacity_BN2 = maxK

        if self._csr_capacity_R < maxR:
            self._stone_pointer_buf = torch.empty(maxR + 1, dtype=torch.int32, device=dev)
            self._group_offset_buf = torch.empty(B + 1, dtype=torch.int32, device=dev)
            self._csr_capacity_R = maxR

        stone_point_ids_buf = self._stone_point_ids_buf
        group_local_id_buf = self._group_local_id_buf
        stone_pointer_buf = self._stone_pointer_buf
        group_offset_buf = self._group_offset_buf

        # 1) Enumerate stones in board point coordinates
        has_stone = self.boards != Stone.EMPTY   # (B,N2)
        stone_board_idx, stone_point_idx = has_stone.nonzero(as_tuple=True)  # (K,), (K,)
        stone_root_idx = roots[stone_board_idx, stone_point_idx]             # (K,) int32

        # 2) Sort by (board, root) so each connected group becomes one CSR segment
        group_sort_key = stone_board_idx.to(torch.int64) * (N2 + 1) + stone_root_idx.to(torch.int64)
        sort_perm = group_sort_key.argsort()

        board_idx_sorted = stone_board_idx[sort_perm]             # (K,)
        point_idx_sorted = stone_point_idx[sort_perm]             # (K,)
        root_idx_sorted = stone_root_idx[sort_perm]               # (K,)
        K = board_idx_sorted.numel()

        # 3) Find CSR run starts for each batch-global group
        same_group_as_prev = (
            (board_idx_sorted == torch.roll(board_idx_sorted, 1))
            & (root_idx_sorted == torch.roll(root_idx_sorted, 1))
        )
        is_group_start = (~same_group_as_prev) | (torch.arange(K, device=dev) == 0)  # (K,) bool

        group_start_offsets = torch.nonzero(is_group_start, as_tuple=True)[0]    # (R,)
        R = group_start_offsets.numel()
        board_idx_per_group = board_idx_sorted[group_start_offsets]               # (R,)
        global_group_index = torch.arange(R, device=dev, dtype=torch.int64)      # (R,)

        global_group_index_for_stone = is_group_start.to(torch.int64).cumsum(0) - 1
        # clamp_min(0): cumsum-1 is -1 only at the very first stone (always a group
        # start), so the bincount slot 0 gets an extra +1 that is harmless because
        # that slot is the first group anyway.
        stones_per_group = torch.bincount(
            global_group_index_for_stone.clamp_min(0),
            minlength=R,
        ).to(torch.int32)                                                          # (R,)

        # 4) Derive board-local group ids from batch-global ones.
        #    global_group_index is 0..R-1 across the whole batch.
        #    Subtracting the first global id of each board gives 0-based local ids.
        groups_per_board = torch.bincount(
            board_idx_per_group.to(torch.int64),
            minlength=B,
        ).to(torch.int32)                                                          # (B,)

        group_offset_by_board = group_offset_buf[:B + 1]
        group_offset_by_board.zero_()
        if B > 0:
            group_offset_by_board[1:] = groups_per_board.cumsum(0)
        first_global_group_per_board = group_offset_by_board[:-1].to(torch.int64)

        local_group_id_per_group = (
            global_group_index
            - first_global_group_per_board.index_select(0, board_idx_per_group)
        ).to(torch.int32)                                                          # (R,)

        # 5) Materialize batch-global CSR arrays and root→local lookup LUT.
        stone_point_ids = stone_point_ids_buf[:K]
        if K > 0:
            stone_point_ids.copy_(point_idx_sorted.to(torch.int32))

        stone_pointer_by_group = stone_pointer_buf[:R + 1]
        stone_pointer_by_group.zero_()
        if R > 0:
            stone_pointer_by_group[1 : R + 1] = stones_per_group.cumsum(0)

        group_local_id_by_root_point = group_local_id_buf   # (B,N2)
        group_local_id_by_root_point.fill_(NO_GROUP)
        if R > 0:
            root_point_idx_per_group = root_idx_sorted[group_start_offsets]       # (R,)
            linear_root_lut_index = (
                board_idx_per_group.to(torch.int64) * N2
                + root_point_idx_per_group.to(torch.int64)
            )
            group_local_id_by_root_point.view(-1).index_put_(
                (linear_root_lut_index,),
                local_group_id_per_group,
                accumulate=False,
            )

        return GroupCsr(
            stone_point_ids=stone_point_ids,
            stone_pointer_by_group=stone_pointer_by_group,
            group_offset_by_board=group_offset_by_board,
            group_local_id_by_root_point=group_local_id_by_root_point,
        )

    # ------------------------------------------------------------------
    # Neighbour helpers (batched point graph, 4 dirs)
    # ------------------------------------------------------------------
    @timed_method
    def _get_neighbor_colors_batch(self) -> Tensor:
        """Return neighbour colours from the board view without per-call index clamps.

        Returns
        -------
        neighbour_colors : (B,N2,4) int8
            For each board, point, direction (N,S,W,E):
            OFF_BOARD_COLOR  : off-board
            Stone.EMPTY      : empty
            Stone.BLACK      : black stone
            Stone.WHITE      : white stone
        """
        B, N2 = self.boards.shape

        # Views only (no alloc): expand precomputed indices & validity
        neighbor_gather_index = self.neighbor_point_ids_safe.view(1, N2, 4).expand(B, -1, -1)
        neighbor_is_on_board = self.neighbor_on_board.view(1, N2, 4).expand(B, -1, -1)

        # Gather from a broadcasted view of the board (no data copy)
        board_by_direction = self.boards.unsqueeze(2).expand(-1, -1, 4)
        out = torch.gather(board_by_direction, dim=1, index=neighbor_gather_index).to(torch.int8)

        # Mark off-board neighbours distinctly
        out.masked_fill_(~neighbor_is_on_board, OFF_BOARD_COLOR)
        return out

    @timed_method
    def _get_neighbor_roots_batch(self, roots: Tensor) -> Tensor:
        """Return neighbour union-find roots using precomputed gather indices; off-board -> NO_ROOT (-1).

        Returns
        -------
        neighbour_roots : (B,N2,4) int32
        """
        B, N2 = roots.shape

        # Views only (no alloc)
        neighbor_gather_index = self.neighbor_point_ids_safe.view(1, N2, 4).expand(B, -1, -1)
        neighbor_is_on_board = self.neighbor_on_board.view(1, N2, 4).expand(B, -1, -1)

        roots_by_direction = roots.unsqueeze(2).expand(-1, -1, 4)
        gathered = torch.gather(roots_by_direction, dim=1, index=neighbor_gather_index)
        gathered.masked_fill_(~neighbor_is_on_board, NO_ROOT)
        return gathered

