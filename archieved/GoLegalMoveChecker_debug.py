# -*- coding: utf-8 -*-
"""
GoLegalMoveChecker.py – board-plane edition (readability pass)
=============================================================

Drop‑in replacement for your existing module. Same public API, clearer names,
explicit shapes, and a batched, **row‑safe** pointer‑jumping step (no cross‑
board leakage). Logic is unchanged except for fixing that bug.

Board representation
--------------------
- board: (B, H, W) int8 with values: -1 empty, 0 black, 1 white
- Internally we often work on the flattened grid graph: N2 = H*W

Notes
-----
- Union‑find uses one vectorised hook sweep + a few per‑row pointer jumps.
- Uses precomputed neighbour tables in flat space.

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union

import torch

# -----------------------------------------------------------------------------
# Dtypes & sentinels
# -----------------------------------------------------------------------------
DTYPE_COLOR: torch.dtype = torch.int16   # stores -1/0/1
IDX_DTYPE:   torch.dtype = torch.int64   # indices / counts

SENTINEL_NEIGH_COLOR = -2  # used for out-of-board neighbour color fills
SENTINEL_NEIGH_ROOT  = -1  # used for out-of-board neighbour root fills

# -----------------------------------------------------------------------------
# Small layout helper for 2D<->1D conversions (readability only)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class BoardLayout:
    B: int
    H: int
    W: int

    @property
    def N2(self) -> int:
        return self.H * self.W

    def flatten(self, x: torch.Tensor) -> torch.Tensor:
        """(B,H,W)->(B,N2) or passthrough for already flat (B,N2)."""
        return x.reshape(self.B, self.N2)

    def unflatten(self, x: torch.Tensor) -> torch.Tensor:
        """(B,N2)->(B,H,W)."""
        return x.reshape(self.B, self.H, self.W)

# =============================================================================
# Public API
# =============================================================================
class GoLegalMoveChecker:
    """Vectorised legal‑move checker with capture detection for Go."""

    def __init__(
        self,
        board_size: int = 19,
        device: Optional[torch.device] = None,
    ):
        self.board_size = board_size
        self.N2         = board_size * board_size
        self.device     = device

        self._checker = VectorizedBoardChecker(board_size, self.device)

    @torch.inference_mode()
    def compute_legal_moves_with_captures(
        self,
        board: torch.Tensor,               # (B, H, W)  int8   –-1/0/1
        current_player: torch.Tensor,      # (B,)        uint8  0/1
        return_capture_info: bool = True,
    ) -> Union[torch.Tensor,
               Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute legal‑move mask (and capture metadata) for each board.

        Returns
        -------
        legal_mask : (B, H, W) bool
        capture_info : dict (only if *return_capture_info* is True)
        """
        B, H, W = board.shape
        assert H == self.board_size and W == self.board_size, "board size mismatch"

        legal_mask, capture_info = self._checker.compute_batch_legal_and_captures(
            board, current_player
        )

        return (legal_mask, capture_info) if return_capture_info else legal_mask


# =============================================================================
# Batch‑vectorised board checker
# =============================================================================
class VectorizedBoardChecker:
    """
    Fully batched legal‑move logic with capture detection.
    Works directly on *(B, H, W)* int8 boards (-1/0/1), but flattens internally
    for graph operations (N2 = H*W).
    """

    # Direction order: N, S, W, E
    OFFSETS_ORDER = ("N", "S", "W", "E")

    def __init__(self, board_size: int, device: torch.device | None):
        self.board_size = board_size
        self.N2         = board_size * board_size
        self.device     = device
        self._init_neighbor_structure()

    # ------------------------------------------------------------------
    # Precomputed neighbours in flat space
    # ------------------------------------------------------------------
    def _init_neighbor_structure(self) -> None:
        N  = self.board_size
        N2 = self.N2
        dev = self.device

        OFF  = torch.tensor([-N, N, -1, 1], dtype=IDX_DTYPE, device=dev)   # (4,)
        flat = torch.arange(N2, dtype=IDX_DTYPE, device=dev)                # (N2,)
        nbrs = flat[:, None] + OFF                                          # (N2,4)

        valid = (nbrs >= 0) & (nbrs < N2)
        col   = flat % N
        valid[:, 2] &= col != 0       # W blocked on left edge
        valid[:, 3] &= col != N - 1   # E blocked on right edge

        self.NEIGH_IDX   = torch.where(valid, nbrs, torch.full_like(nbrs, -1))  # (N2,4)
        self.NEIGH_VALID = valid                                                 # (N2,4) bool

    # ------------------------------------------------------------------
    # Top‑level batch computation
    # ------------------------------------------------------------------
    def compute_batch_legal_and_captures(
        self,
        board: torch.Tensor,          # (B, H, W)  int8  –-1/0/1
        current_player: torch.Tensor  # (B,)        uint8 0/1
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, H, W = board.shape
        layout = BoardLayout(B, H, W)
        N2 = layout.N2

        board_f  = layout.flatten(board)             # (B, N2) int8
        empty    = (board_f == -1)                   # (B, N2) bool

        # Union‑find (groups) & liberties per group
        _parent, colour, roots, root_libs = self._batch_init_union_find(board_f)

        # Neighbour lookups (colour/roots) in flat space, with valid mask
        neigh_colors = self._get_neighbor_colors_batch(colour)   # (B,N2,4)
        neigh_roots  = self._get_neighbor_roots_batch(roots)     # (B,N2,4)
        valid_mask   = self.NEIGH_VALID.view(1, N2, 4)

        curr = current_player.view(B, 1, 1)  # (B,1,1)
        opp  = 1 - curr

        # Immediate liberties if we play here (pre‑move): any adjacent empty?
        has_any_lib = ((neigh_colors == -1) & valid_mask).any(dim=2)       # (B,N2)

        # For each neighbour, gather that group's liberty count
        neigh_roots_f = neigh_roots.reshape(B, -1)                         # (B,N2*4)
        neigh_libs_f  = root_libs.gather(1, neigh_roots_f.clamp(min=0))    # (B,N2*4)
        neigh_libs    = neigh_libs_f.view(B, N2, 4)                        # (B,N2,4)

        # Captures: neighbouring opponent group with exactly 1 liberty
        opp_mask        = (neigh_colors == opp) & valid_mask               # (B,N2,4)
        can_capture     = opp_mask & (neigh_libs == 1)                     # (B,N2,4)
        can_capture_any = can_capture.any(dim=2)                            # (B,N2)

        # Friendly attachment that is safe immediately (>1 liberties pre‑move)
        friendly        = (neigh_colors == curr) & valid_mask              # (B,N2,4)
        friendly_any    = (friendly & (neigh_libs > 1)).any(dim=2)         # (B,N2)

        # Final legality: empty AND (has_lib OR captures OR safe_friendly)
        legal_flat = empty & (has_any_lib | can_capture_any | friendly_any)
        legal_mask = legal_flat.view(B, H, W)

        # ----- Capture meta‑data for engine (roots to delete, and sizes) -----
        capture_groups = torch.full((B, N2, 4), SENTINEL_NEIGH_ROOT,
                                    dtype=IDX_DTYPE, device=self.device)
        capture_groups[can_capture] = neigh_roots[can_capture]             # (B,N2,4)

        # Group sizes (count stones per root id)
        sizes = torch.zeros((B, N2), dtype=IDX_DTYPE, device=self.device)
        sizes.scatter_add_(1, roots, torch.ones_like(roots, dtype=IDX_DTYPE))

        # For captured neighbour roots, attach their sizes per direction
        capture_sizes = torch.zeros_like(capture_groups)
        valid_cap     = capture_groups >= 0
        cap_flat      = capture_groups.view(B, -1)
        valid_flat    = valid_cap.view(B, -1)

        sizes_flat    = sizes.gather(1, cap_flat.clamp(min=0))
        capture_sizes.view(B, -1)[valid_flat] = sizes_flat[valid_flat]

        total_captures = capture_sizes.sum(dim=2).view(B, H, W)             # (B,H,W)

        capture_info: Dict[str, torch.Tensor] = {
            "would_capture" : (legal_flat & can_capture_any).view(B, H, W),
            "capture_groups": capture_groups.view(B, H, W, 4),
            "capture_sizes" : capture_sizes.view(B, H, W, 4),
            "total_captures": total_captures,
            "roots"         : roots,     # (B,N2)
            "colour"        : colour,    # (B,N2)
        }
        return legal_mask, capture_info

    # ------------------------------------------------------------------
    # Neighbour helpers (batched, flat graph)
    # ------------------------------------------------------------------
    def _get_neighbor_colors_batch(self, colour: torch.Tensor) -> torch.Tensor:
        B, N2 = colour.shape
        idx   = self.NEIGH_IDX.view(1, N2, 4).expand(B, -1, -1)            # (B,N2,4)
        valid = self.NEIGH_VALID.view(1, N2, 4).expand(B, -1, -1)          # (B,N2,4)

        idx_flat   = idx.reshape(B, -1)
        valid_flat = valid.reshape(B, -1)

        out = torch.full_like(idx_flat, SENTINEL_NEIGH_COLOR, dtype=DTYPE_COLOR)
        gathered = torch.gather(colour, 1, idx_flat.clamp(min=0))
        out[valid_flat] = gathered[valid_flat]
        return out.view(B, N2, 4)

    def _get_neighbor_roots_batch(self, roots: torch.Tensor) -> torch.Tensor:
        B, N2 = roots.shape
        idx   = self.NEIGH_IDX.view(1, N2, 4).expand(B, -1, -1)            # (B,N2,4)
        valid = self.NEIGH_VALID.view(1, N2, 4).expand(B, -1, -1)          # (B,N2,4)

        idx_flat   = idx.reshape(B, -1)
        valid_flat = valid.reshape(B, -1)

        out = torch.full_like(idx_flat, SENTINEL_NEIGH_ROOT, dtype=IDX_DTYPE)
        gathered = torch.gather(roots, 1, idx_flat.clamp(min=0))
        out[valid_flat] = gathered[valid_flat]
        return out.view(B, N2, 4)

    # ------------------------------------------------------------------
    # _hook_and_compress
    # ------------------------------------------------------------------

    def _hook_and_compress(self, parent: torch.Tensor, same: torch.Tensor) -> torch.Tensor:
        B, N2 = parent.shape
        dev   = parent.device
        nbr_idx  = self.NEIGH_IDX.view(1, N2, 4).expand(B, -1, -1)
        max_rounds = int((N2 - 1).bit_length())

        for _ in range(max_rounds):
            parent_prev = parent
            nbr_parent = torch.gather(parent, 1, nbr_idx.clamp(min=0).reshape(B, -1)).view(B, N2, 4)
            nbr_parent = torch.where(same, nbr_parent, torch.full_like(nbr_parent, N2))
            min_nbr_parent = nbr_parent.min(dim=2).values
            hooked = torch.minimum(parent, min_nbr_parent)
            comp   = torch.gather(hooked, 1, hooked)
            comp2  = torch.gather(comp,   1, comp)
            if torch.equal(comp2, parent_prev):
                return comp2
            parent = comp2
        return parent
    
    # ------------------------------------------------------------------
    # Union‑find + liberties (flat graph; **row‑safe** compression)
    # ------------------------------------------------------------------
    def _batch_init_union_find(
        self,
        board_f: torch.Tensor  # (B, N2) int8 in {-1,0,1}
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N2 = board_f.shape
        dev = self.device

        # Colours in fixed dtype (no changes to values)
        colour = board_f.to(DTYPE_COLOR)                                    # (B,N2)

        # Start with identity parents per row (local ids 0..N2-1)
        parent = torch.arange(N2, dtype=IDX_DTYPE, device=dev) \
                     .expand(B, N2).clone()                                # (B,N2)

        # Same‑colour adjacency mask
        neigh_cols = self._get_neighbor_colors_batch(colour)                # (B,N2,4)
        same = (neigh_cols == colour.unsqueeze(2)) \
               & (colour.unsqueeze(2) != -1) \
               & self.NEIGH_VALID.view(1, N2, 4)                            # (B,N2,4)

        # Replace the one-shot hook with iterative relaxation
        parent = torch.arange(N2, dtype=IDX_DTYPE, device=dev).expand(B, N2).clone()

        parent = self._hook_and_compress(parent, same)
        roots = parent.clone()                                              # (B,N2)

        # ---- Count unique liberties per root ----
        neigh_cols2 = self._get_neighbor_colors_batch(colour)               # (B,N2,4)
        is_lib      = (neigh_cols2 == -1) & self.NEIGH_VALID.view(1, N2, 4) # (B,N2,4)
        stone_mask  = (colour != -1)                                        # (B,N2)

        libs_per_root = torch.zeros(B * N2, dtype=IDX_DTYPE, device=dev)    # (B*N2,)

        if stone_mask.any():
            batch_map = torch.arange(B, dtype=IDX_DTYPE, device=dev).view(B, 1, 1)
            roots_exp = roots.unsqueeze(2)                                  # (B,N2,1)
            lib_idx   = self.NEIGH_IDX.view(1, N2, 4)                       # (1,N2,4)
            mask      = is_lib & stone_mask.unsqueeze(2)                    # (B,N2,4)

            fb = batch_map.expand_as(mask)[mask]                            # (K,)
            fr = roots_exp.expand_as(mask)[mask]                            # (K,)
            fl = lib_idx.expand_as(mask)[mask]                              # (K,)

            key_root = fb * N2 + fr
            key_lib  = fb * N2 + fl
            pairs    = torch.stack((key_root, key_lib), dim=1)              # (K,2)

            # DEBUG: Print pairs before deduplication for Root 0
            if B == 1 and N2 == 25:  # Only for our test case (5x5 board, batch=1)
                # Find pairs for root 0
                root_0_pairs = pairs[pairs[:, 0] == 0]  # key_root == 0 means batch 0, root 0
                if root_0_pairs.shape[0] > 0:
                    print("\n[DEBUG] Liberty pairs for Root 0 before deduplication:")
                    for i, (root_key, lib_key) in enumerate(root_0_pairs):
                        lib_idx_actual = lib_key.item() % N2
                        print(f"  Pair {i}: root_key={root_key.item()}, lib_key={lib_key.item()}, liberty_idx={lib_idx_actual}")

            # Deduplicate by (root, liberty point)
            pairs = pairs[pairs[:, 1].argsort()]
            uniq  = torch.unique_consecutive(pairs, dim=0)

            # DEBUG: Print unique pairs for Root 0
            if B == 1 and N2 == 25:
                root_0_uniq = uniq[uniq[:, 0] == 0]
                if root_0_uniq.shape[0] > 0:
                    print(f"\n[DEBUG] Unique liberty pairs for Root 0 after deduplication:")
                    for i, (root_key, lib_key) in enumerate(root_0_uniq):
                        lib_idx_actual = lib_key.item() % N2
                        lib_row, lib_col = lib_idx_actual // 5, lib_idx_actual % 5
                        print(f"  Unique pair {i}: root_key={root_key.item()}, lib_key={lib_key.item()}, liberty at [{lib_row},{lib_col}]")
                    print(f"  Total unique liberties for Root 0: {root_0_uniq.shape[0]}")

            libs_per_root.scatter_add_(0, uniq[:, 0],
                                       torch.ones_like(uniq[:, 0], dtype=IDX_DTYPE))
            
            # DEBUG: Check what scatter_add actually did
            if B == 1 and N2 == 25:
                print(f"\n[DEBUG] libs_per_root[0] after scatter_add: {libs_per_root[0].item()}")

        root_libs = libs_per_root.reshape(B, N2)                            # (B,N2)
        
        # DEBUG: Final check
        if B == 1 and N2 == 25:
            print(f"[DEBUG] Final root_libs[0, 0]: {root_libs[0, 0].item()}")
        return parent, colour, roots, root_libs
