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

from functools import wraps

def _mem_alloc_mb(dev: torch.device) -> float:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        return torch.cuda.memory_allocated(dev) / (1024**2)
    if dev.type == "mps" and hasattr(torch, "mps"):
        return torch.mps.current_allocated_memory() / (1024**2)
    return 0.0

def _mem_aux_str(dev: torch.device) -> str:
    if dev.type == "cuda":
        peak = torch.cuda.max_memory_allocated(dev) / (1024**2)
        return f"peak={peak:.1f} MB"
    if dev.type == "mps" and hasattr(torch, "mps"):
        drv = torch.mps.driver_allocated_memory() / (1024**2)
        return f"driver={drv:.1f} MB"
    return ""

def mem_probe(tag: str, min_delta_mb: float = 8.0):
    """Function decorator: prints net allocation delta if |Δ| >= threshold."""
    def deco(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            dev = getattr(self, "device", torch.device("cpu"))
            before = _mem_alloc_mb(dev)
            out = fn(self, *args, **kwargs)
            after = _mem_alloc_mb(dev)
            delta = after - before
            if abs(delta) >= min_delta_mb:
                aux = _mem_aux_str(dev)
                print(f"[MEM][{tag}] delta={delta:+.1f} MB alloc={after:.1f} MB {aux}")
            return out
        return wrapper
    return deco



# =============================================================================
# Public API
# =============================================================================
class GoLegalMoveChecker:
    def __init__(self, board_size=19, device=None):
        self._checker = VectorizedBoardChecker(board_size, device)

    def compute_batch_legal_and_info(self, board, current_player, return_info=True):
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
        self.N2         = board_size * board_size
        self.device     = device
        self._init_board_fallten_structure()

    # ------------------------------------------------------------------
    # Precomputed 4-neighbour tables in flat space
    # ------------------------------------------------------------------
    def _init_board_fallten_structure(self) -> None:
        N, N2, dev = self.board_size, self.N2, self.device

        # (N2,) flat indices of a single board
        self.index_flatten = torch.arange(N2, dtype=torch.int64, device=dev)

        # (N2,4) neighbours via offsets: N,S,W,E
        OFF  = torch.tensor([-N, N, -1, 1], dtype=torch.int64, device=dev)    # (4,)
        nbrs = self.index_flatten[:, None] + OFF                               # (N2,4)

        # Edge handling
        valid = (nbrs >= 0) & (nbrs < N2)                                      # (N2,4)
        col   = self.index_flatten % N
        valid[:, 2] &= col != 0           # W invalid at left edge
        valid[:, 3] &= col != N - 1       # E invalid at right edge

        self.neigh_index_flatten = torch.where(valid, nbrs, torch.full_like(nbrs, -1))
        self.neigh_valid_flatten = valid
        # Non-negative neighbor indices (off-board → 0) for gather; precomputed once.
        self.neigh_index_nonneg_flatten = torch.where(
        self.neigh_valid_flatten, self.neigh_index_flatten, torch.zeros_like(self.neigh_index_flatten)
        )  # (N2,4) int64
        


    # ------------------------------------------------------------------
    # Top-level: legal mask + capture metadata (CSR-based; no dense BxN2xN2)
    # ------------------------------------------------------------------
    def compute_batch_legal_and_info(
        self,
        board: torch.Tensor,          # (B,H,W) int8 in {-1,0,1}
        current_player: torch.Tensor  # (B,)    uint8 in {0,1}
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, H, W = board.shape
        N2      = self.N2
        dev     = self.device

        # Per-call runtime flatten (depends on B)
        self.board_flatten = board.reshape(B, N2)         # (B,N2)
        empty              = (self.board_flatten == -1)   # (B,N2) bool

        # Groups (roots) + liberties
        roots, root_libs = self._batch_init_union_find()  # (B,N2) each

        # === Build the CSR + LUTs (batch-wide) ==================================
        csr = self._build_group_csr(roots)
        stone_global_index              = csr["stone_global_index"]               # (K,)
        stone_global_pointer            = csr["stone_global_pointer"]             # (R+1,)
        group_global_pointer_per_board  = csr["group_global_pointer_per_board"]   # (B+1,)
        stone_local_index_from_cell     = csr["stone_local_index_from_cell"]      # (B,N2)
        stone_local_index_from_root     = csr["stone_local_index_from_root"]      # (B,N2)

        # === Neighbour tables ====================================================
        neigh_valid_flatten_b = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)
        neigh_colors = self._get_neighbor_colors_batch()             # (B,N2,4)
        neigh_roots  = self._get_neighbor_roots_batch(roots)         # (B,N2,4)

        curr = current_player.view(B, 1, 1)  # (B,1,1)
        opp  = 1 - curr

        # A) immediate liberties
        has_any_lib = ((neigh_colors == -1) & neigh_valid_flatten_b).any(dim=2)   # (B,N2)

        # B) captures: adjacent opponent group with exactly 1 liberty
        neigh_roots_f = neigh_roots.reshape(B, -1)                                 # (B,N2*4)
        neigh_libs_f  = root_libs.gather(1, neigh_roots_f.clamp(min=0))            # (B,N2*4)
        neigh_libs    = neigh_libs_f.view(B, N2, 4)                                 # (B,N2,4)

        opp_mask        = (neigh_colors == opp) & neigh_valid_flatten_b             # (B,N2,4) bool
        can_capture     = opp_mask & (neigh_libs == 1)                              # (B,N2,4)
        can_capture_any = can_capture.any(dim=2)                                    # (B,N2)

        # C) friendly safe attachment
        friendly     = (neigh_colors == curr) & neigh_valid_flatten_b               # (B,N2,4)
        friendly_any = (friendly & (neigh_libs > 1)).any(dim=2)                     # (B,N2)

        # D) final legality
        legal_flat = empty & (has_any_lib | can_capture_any | friendly_any)
        legal_mask = legal_flat.view(B, H, W)                                       # (B,H,W)

        # ----- Capture metadata (CSR-based; no dense mask) -----------------------
        # Map neighbour *roots* → local group ids (or -1); keep only capturing dirs
        captured_group_local_index_all = stone_local_index_from_root.gather(
            1, neigh_roots.clamp(min=0).reshape(B, -1).to(torch.int64)
        ).view(B, N2, 4)                                                            # (B,N2,4) int32

        captured_group_local_index = torch.where(
            can_capture,
            captured_group_local_index_all,
            torch.full_like(captured_group_local_index_all, -1, dtype=torch.int32)
        )                                                                           # (B,N2,4) int32

        # info payload (no (B,N2,N2) tensors; use CSR + per-candidate gids)
        info: Dict[str, torch.Tensor] = {
            # core group topology
            "roots": roots,                          # (B,N2) int64
            "root_libs": root_libs,                  # (B,N2) int64

            # legality helpers
            "can_capture_any": can_capture_any,      # (B,N2) bool
            "captured_group_local_index": captured_group_local_index.long(),  # (B,N2,4) int32

            # CSR (global, batch-wide)
            "stone_global_index":             stone_global_index.long(),              # (K,)   int32
            "stone_global_pointer":           stone_global_pointer.long(),            # (R+1,) int32
            "group_global_pointer_per_board": group_global_pointer_per_board.long(),  # (B+1,) int32

            # LUTs (fast ID→ID maps)
            "stone_local_index_from_cell":    stone_local_index_from_cell.long(),     # (B,N2) int32
            "stone_local_index_from_root":    stone_local_index_from_root.long(),     # (B,N2) int32
        }

        return legal_mask, info

    @mem_probe("UF + liberties")   # ⬅️ add this line
    def _batch_init_union_find(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        roots     : (B,N2) int32  union-find representative per point
        root_libs : (B,N2) int32  liberty count per root id (index by root id)
        """
        board_flatten = self.board_flatten
        B, N2 = board_flatten.shape
        dev   = self.device

        # ---- Neighbour colors (from board_flatten) --------------------------------
        neigh_valid_flatten_b = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)
        neigh_cols            = self._get_neighbor_colors_batch()  # (B,N2,4)

        # ---- Same-color adjacency (ignore empties; respect edges) -----------------
        same = (neigh_cols == board_flatten.unsqueeze(2)) \
            & (board_flatten.unsqueeze(2) != -1) \
            & neigh_valid_flatten_b                                              # (B,N2,4)

        # ---- Hook & compress (union-find) ----------------------------------------
        parent = torch.arange(N2, dtype=torch.int32, device=dev).expand(B, N2)    # (B,N2)
        parent = self._hook_and_compress(parent, same)
        roots  = parent                                                           # (B,N2)

        # ---- Count unique liberties per root -------------------------------------
        is_lib     = (neigh_cols == -1) & neigh_valid_flatten_b                   # (B,N2,4)
        stone_mask = (board_flatten != -1)                                        # (B,N2)
        mask       = is_lib & stone_mask.unsqueeze(2)                             # (B,N2,4)

        # K = number of stone→empty edges across the batch
        fb, fj, fd = torch.where(mask)            # each (K,)
        fr = roots[fb, fj]        # (K,)  root id of the stone
        fl = self.neigh_index_flatten[fj, fd] # (K,) liberty cell id (valid)

        # Deduplicate by (batch, root, liberty_point) then count uniques per root
        key_root = fb * N2 + fr                   # (K,)
        key_lib  = fb * N2 + fl                   # (K,)
        pairs    = torch.stack((key_root, key_lib), dim=1)        # (K,2)

        sort_key     = pairs[:, 0].to(torch.int64) * (N2 * B) + pairs[:, 1].to(torch.int64)
        sorted_idx   = sort_key.argsort()
        pairs_sorted = pairs[sorted_idx]
        uniq         = torch.unique_consecutive(pairs_sorted, dim=0)              # (Kuniq,2)

        libs_per_root = torch.zeros(B * N2, dtype=torch.int32, device=dev)        # (B*N2,)
        if uniq.numel() > 0:
            libs_per_root.scatter_add_(
                0,
                uniq[:, 0],
                torch.ones(uniq.size(0), dtype=torch.int32, device=dev)
            )

        root_libs = libs_per_root.view(B, N2)                                     # (B,N2)
        return roots, root_libs


    # ------------------------------------------------------------------
    # Batched pointer-jumping with periodic convergence check (int32)
    # ------------------------------------------------------------------
    def _hook_and_compress(self, parent: torch.Tensor, same: torch.Tensor) -> torch.Tensor:
        B, N2 = parent.shape

        # Keep gather indices as long (int64); clamp off-board to 0 and mask later.
        nbr_idx  = self.neigh_index_flatten.view(1, N2, 4).expand(B, -1, -1).clamp(min=0)  # long
        idx_flat = nbr_idx.reshape(B, -1)

        max_rounds = (N2).bit_length() + 4  # ~log2(N2)+slack

        for i in range(max_rounds):
            # parents of 4 neighbours (parent is int32, indices are long)
            nbr_parent = torch.gather(parent, 1, idx_flat).view(B, N2, 4)  # int32
            nbr_parent.masked_fill_(~same, N2)  # sentinel >= N2 so it won't be chosen

            # hook to smallest neighbour representative
            min_nbr = nbr_parent.min(dim=2).values            # (B,N2) int32
            hooked  = torch.minimum(parent, min_nbr)          # (B,N2) int32

            # ONE pointer jump (compression)
            parent_next = torch.gather(hooked, 1, hooked.long())  # (B,N2) int32

            # lazy early-exit (every 4 rounds) to limit device syncs
            if (i & 3) == 3 and torch.equal(parent_next, parent):
                return parent_next

            parent = parent_next

        return parent
    
    
    # def _mem_report(self, tag: str) -> None:
    #     dev = self.device
    #     try:
    #         if dev.type == "cuda":
    #             torch.cuda.synchronize(dev)
    #             cur = torch.cuda.memory_allocated(dev) / (1024**2)
    #             peak = torch.cuda.max_memory_allocated(dev) / (1024**2)
    #             print(f"[{tag}][cuda] alloc={cur:.1f} MB peak={peak:.1f} MB")
    #         elif dev.type == "mps" and hasattr(torch, "mps"):
    #             # MPS counters are per-process; no device arg
    #             cur = torch.mps.current_allocated_memory() / (1024**2)
    #             drv = torch.mps.driver_allocated_memory() / (1024**2)
    #             print(f"[{tag}][mps] alloc={cur:.1f} MB driver={drv:.1f} MB")
    #         else:
    #             # CPU fallback
    #             print(f"[{tag}][{dev.type}] (no device mem stats)")
    #     except Exception as e:
    #         print(f"[{tag}] mem stats unavailable: {e}")

    # @torch.no_grad()
    # def _hook_and_compress(self, parent: torch.Tensor, same: torch.Tensor) -> torch.Tensor:
    #     B, N2 = parent.shape

    #     # views only (no alloc)
    #     nbr_idx  = self.neigh_index_flatten.view(1, N2, 4).expand(B, -1, -1).clamp(min=0)
    #     idx_flat = nbr_idx.reshape(B, -1)

    #     # optional: reset CUDA peak stats at function entry
    #     if self.device.type == "cuda":
    #         torch.cuda.reset_peak_memory_stats(self.device)

    #     max_rounds = (N2).bit_length() + 4  # ~log2(N2)+slack
    #     approx_mb_nbr = (B * N2 * 4 * 4) / 1e6  # (B,N2,4) * 4 bytes → MB

    #     prev_ptr = None
    #     rounds = 0

    #     for i in range(max_rounds):
    #         # BIG alloc this round
    #         nbr_parent = torch.gather(parent, 1, idx_flat).view(B, N2, 4)  # int32
    #         ptr = int(nbr_parent.data_ptr())
    #         reused = (ptr == prev_ptr)
    #         prev_ptr = ptr

    #         # mask off non-same edges
    #         nbr_parent.masked_fill_(~same, N2)

    #         # standard UF step
    #         min_nbr     = nbr_parent.min(dim=2).values            # (B,N2)
    #         hooked      = torch.minimum(parent, min_nbr)          # (B,N2)
    #         parent_next = torch.gather(hooked, 1, hooked.long())  # (B,N2)

    #         rounds += 1

    #         # print every 4 rounds to reduce noise (change as needed)
    #         if (i & 3) == 3 or i == 0:
    #             self._mem_report(f"UF iter {i}")
    #             print(f"[UF] iter={i:02d} nbr_parent.ptr={ptr} "
    #                 f"reused_vs_prev={reused} approx_size={approx_mb_nbr:.1f} MB "
    #                 f"B={B} N2={N2}")

    #         # early exit check
    #         if (i & 3) == 3 and torch.equal(parent_next, parent):
    #             self._mem_report("UF converged")
    #             print(f"[UF] converged in {rounds} rounds; "
    #                 f"peak big temp ≈ (B,N2,4) int32 ~ {approx_mb_nbr:.1f} MB")
    #             return parent_next

    #         parent = parent_next

    #     self._mem_report("UF max-rounds")
    #     print(f"[UF] hit max rounds {rounds}; "
    #         f"peak big temp ≈ (B,N2,4) int32 ~ {approx_mb_nbr:.1f} MB")
    #     return parent

    # ------------------------------------------------------------------
    # Build CSR + LUTs (global across batch; safe for K=0)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _build_group_csr(self, roots: torch.Tensor):
        dev = self.device
        B, N2 = self.board_flatten.shape

        # 1) take stones (exclude empties)
        is_stone = (self.board_flatten != -1)  # (B,N2)
        # 1) take stones (exclude empties)
        b_all, j_all = is_stone.nonzero(as_tuple=True)                          # (K,), (K,)
        r_all = roots[b_all, j_all]                                             # (K,)


        # 2) global sort by (board, root) → stones contiguous per group
        sort_key = b_all * (N2 + 1) + r_all
        perm = sort_key.argsort()
        b_sorted = b_all[perm]         # (K,)
        j_sorted = j_all[perm]         # (K,)
        r_sorted = r_all[perm]         # (K,)

        K = b_sorted.numel()

        # 3) run boundaries for (board, root) (no special-case at 0)
        same_prev = (b_sorted == torch.roll(b_sorted, 1)) & (r_sorted == torch.roll(r_sorted, 1))
        new_group = (~same_prev) | (torch.arange(K, device=dev) == 0)             # (K,) bool

        run_starts = torch.nonzero(new_group, as_tuple=True)[0]                   # (R,)
        R = run_starts.numel()
        run_board = b_sorted[run_starts]                                          # (R,)
        run_idx = torch.arange(R, device=dev, dtype=torch.int64)                  # (R,)

        run_id_for_stone = new_group.to(torch.int64).cumsum(0) - 1                # (K,)
        run_sizes = torch.bincount(run_id_for_stone.clamp_min(0), minlength=R).to(torch.int32)  # (R,)

        # 4) board pointers (local→global bridge)
        groups_per_board = torch.bincount(run_board.to(torch.int64), minlength=B).to(torch.int32)  # (B,)
        group_global_pointer_per_board = torch.zeros(B + 1, dtype=torch.int32, device=dev)         # (B+1,)
        group_global_pointer_per_board[1:] = groups_per_board.cumsum(0)
        board_first_global = group_global_pointer_per_board[:-1].to(torch.int64)                   # (B,)

        # local gid per run
        gid_of_run_local = (run_idx - board_first_global.index_select(0, run_board)).to(torch.int32)  # (R,)
        gid_for_stone_local = gid_of_run_local[run_id_for_stone]                                       # (K,)

        # 5) outputs (CSR arrays + LUTs)
        stone_global_index   = j_sorted.to(torch.int32)                                   # (K,)
        stone_global_pointer = torch.zeros(R + 1, dtype=torch.int32, device=dev)         # (R+1,)
        if R > 0:
            stone_global_pointer[1:] = run_sizes.cumsum(0)

        # (b, cell) -> local gid  (scatter on flat buffer)
        stone_local_index_from_cell = torch.full((B, N2), -1, dtype=torch.int32, device=dev)
        lin_cells = (b_sorted * N2 + j_sorted).to(torch.int64)                            # (K,)
        if K > 0:
            stone_local_index_from_cell.view(-1).index_put_(
                (lin_cells,), gid_for_stone_local, accumulate=False
            )

        # (b, UF root) -> local gid
        stone_local_index_from_root = torch.full((B, N2), -1, dtype=torch.int32, device=dev)
        root_id_of_run = r_sorted[run_starts]                                             # (R,)
        if R > 0:
            lin_roots = (run_board * N2 + root_id_of_run).to(torch.int64)                 # (R,)
            stone_local_index_from_root.view(-1).index_put_(
                (lin_roots,), gid_of_run_local, accumulate=False
            )

        return {
            "stone_global_index":              stone_global_index,              # (K,)
            "stone_global_pointer":            stone_global_pointer,            # (R+1,)
            "group_global_pointer_per_board":  group_global_pointer_per_board,  # (B+1,)
            "stone_local_index_from_cell":     stone_local_index_from_cell,     # (B,N2)
            "stone_local_index_from_root":     stone_local_index_from_root,     # (B,N2)
        }

    # ------------------------------------------------------------------
    # Batched pointer-jumping with periodic convergence check
    # ------------------------------------------------------------------
    # def _hook_and_compress(self, parent: torch.Tensor, same: torch.Tensor) -> torch.Tensor:
    #     B, N2 = parent.shape
    #     neigh_index_flatten_b = self.neigh_index_flatten.view(1, N2, 4).expand(B, -1, -1)
    #     max_rounds = N2

    #     for i in range(max_rounds):
    #         parent_prev = parent
    #         nbr_parent  = torch.gather(parent, 1, neigh_index_flatten_b.clamp(min=0).reshape(B, -1)).view(B, N2, 4) # (B,N2,4)
    #         nbr_parent  = torch.where(same, nbr_parent, torch.full_like(nbr_parent, N2)) # (B,N2,4)
    #         min_nbr     = nbr_parent.min(dim=2).values                                   # (B,N2)

    #         hooked = torch.minimum(parent, min_nbr)                                      # (B,N2)
    #         comp   = torch.gather(hooked, 1, hooked)                                     # parent[parent]
    #         comp   = torch.gather(comp,   1, comp)                                       # parent[parent[parent]]
    #         # lazy convergence check
    #         if (i & 3) == 3 and torch.equal(comp, parent_prev):                          # early exit every 4 iters
    #             return comp
    #         parent = comp
    #     return parent

    # ------------------------------------------------------------------
    # Neighbour helpers (batched, flat graph, 4 dirs)
    # ------------------------------------------------------------------
    def _get_neighbor_colors_batch(self) -> torch.Tensor:
        """Return neighbor colors pulled from self.board_flatten without per-call index clamps."""
        B, N2 = self.board_flatten.shape

        # Views only (no alloc): expand precomputed indices & validity
        idx   = self.neigh_index_nonneg_flatten.view(1, N2, 4).expand(B, -1, -1)  # (B,N2,4) long (view)
        valid = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)         # (B,N2,4) bool (view)

        # Gather from a broadcasted view of the board (no data copy)
        board3 = self.board_flatten.unsqueeze(2).expand(-1, -1, 4)                # (B,N2,4) view
        out = torch.gather(board3, dim=1, index=idx).to(torch.int8)               # (B,N2,4) int8

        # Mark off-board neighbors distinctly
        out.masked_fill_(~valid, -2)
        return out


    def _get_neighbor_roots_batch(self, roots: torch.Tensor) -> torch.Tensor:
        """Return neighbor union-find roots using precomputed non-negative indices; off-board = -1."""
        B, N2 = roots.shape

        # Views only (no alloc)
        idx   = self.neigh_index_nonneg_flatten.view(1, N2, 4).expand(B, -1, -1)  # (B,N2,4) long (view)
        valid = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)         # (B,N2,4) bool (view)

        roots3   = roots.unsqueeze(2).expand(-1, -1, 4)                            # (B,N2,4) view
        gathered = torch.gather(roots3, dim=1, index=idx)                          # (B,N2,4), same dtype as roots
        gathered.masked_fill_(~valid, -1)
        return gathered

