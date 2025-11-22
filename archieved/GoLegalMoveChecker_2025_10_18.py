# -*- coding: utf-8 -*-
"""
GoLegalMoveChecker.py – board-plane edition (readability pass)
=============================================================

Drop-in replacement for your existing module. Same public API, clearer names,
explicit shapes, and a batched, row-safe pointer-jumping step.

Board
-----
- board: (B, H, W) int8 with values: -1 empty, 0 black, 1 white
- Internally we work on a flattened grid: N2 = H * W
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Union
import torch

# -----------------------------------------------------------------------------
# Dtypes & sentinels
# -----------------------------------------------------------------------------
DTYPE_COLOR: torch.dtype = torch.int8
IDX_DTYPE:   torch.dtype = torch.int64

SENTINEL_NEIGH_COLOR = -2  # off-board neighbour color fill
SENTINEL_NEIGH_ROOT  = -1  # off-board neighbour root fill


# =============================================================================
# Public API
# =============================================================================
class GoLegalMoveChecker:
    def __init__(self, board_size=19, device=None):
        self._checker = VectorizedBoardChecker(board_size, device)

    def compute_batch_legal_and_info(self, board, current_player, return_info=True):
        B,H,W = board.shape
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

    # Per-board (flatten) structures
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
        self.index_flatten = torch.arange(N2, dtype=IDX_DTYPE, device=dev)

        # (N2,4) neighbours via offsets: N,S,W,E
        OFF  = torch.tensor([-N, N, -1, 1], dtype=IDX_DTYPE, device=dev)    # (4,)
        nbrs = self.index_flatten[:, None] + OFF                             # (N2,4)

        # Edge handling
        valid = (nbrs >= 0) & (nbrs < N2)                                    # (N2,4)
        col   = self.index_flatten % N
        valid[:, 2] &= col != 0           # W invalid at left edge
        valid[:, 3] &= col != N - 1       # E invalid at right edge

        self.neigh_index_flatten = torch.where(valid, nbrs, torch.full_like(nbrs, -1))
        self.neigh_valid_flatten = valid

    # ------------------------------------------------------------------
    # Top-level: legal mask + capture metadata
    # ------------------------------------------------------------------
    def compute_batch_legal_and_info(
        self,
        board: torch.Tensor,          # (B,H,W) int8 in {-1,0,1}
        current_player: torch.Tensor  # (B,)    uint8 in {0,1}
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, H, W = board.shape
        N2      = self.N2

        # Per-call runtime flatten (depends on B)
        self.board_flatten = board.reshape(B, N2)         # (B,N2)
        empty              = (self.board_flatten == -1)   # (B,N2) bool

        # Groups (roots) + liberties   (pass board_flatten instead of “colour”)
        roots, root_libs = self._batch_init_union_find()  # (B,N2) each

        # Batched neighbour tables
        neigh_valid_flatten_b = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)

        # Neighbour colors/roots (colors are read from self.board_flatten)
        neigh_colors = self._get_neighbor_colors_batch()  # (B,N2,4)
        neigh_roots  = self._get_neighbor_roots_batch(roots)  # (B,N2,4)

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

        # ----- Capture metadata: exact stones removed for each play -----
        capture_groups = torch.full(
            (B, N2, 4), SENTINEL_NEIGH_ROOT, dtype=IDX_DTYPE, device=self.device
        )                                                                           # (B,N2,4)
        capture_groups = torch.where(can_capture, neigh_roots, capture_groups)

        # Compare candidate capture roots vs all roots to mark stones
        capture_groups_exp = capture_groups.view(B, N2, 4, 1)                       # (B,N2,4,1)
        roots_exp          = roots.view(B, 1, 1, N2)                                # (B,1,1,N2)
        group_matches      = (capture_groups_exp == roots_exp) & (capture_groups_exp >= 0)
        capture_stone_mask = group_matches.any(dim=2)                               # (B,N2,N2)

        # Keep opponent stones only
        opp_colour         = 1 - current_player.view(B, 1)
        is_opponent_stone  = (self.board_flatten == opp_colour)                      # (B,N2)
        capture_stone_mask = capture_stone_mask & is_opponent_stone.view(B, 1, N2)

        # Belt-and-braces: cannot "capture" the just-placed stone itself here
        diag = torch.arange(N2, device=self.device)
        capture_stone_mask[:, diag, diag] = False

        info: Dict[str, torch.Tensor] = {
            "roots": roots,                          # (B,N2) int64
            "root_libs": root_libs,                 # (B,N2) int64
            "can_capture_any": can_capture_any,     # (B,N2) bool
            "capture_stone_mask": capture_stone_mask,  # (B,N2,N2) bool
        }
        return legal_mask, info

    # ------------------------------------------------------------------
    # Union-find + liberties (flat graph; row-safe compression)
    # ------------------------------------------------------------------
    def _batch_init_union_find(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        roots     : (B,N2) int64  union-find representative per point
        root_libs : (B,N2) int64  liberty count per root id (index by root id)
        """
        board_flatten=self.board_flatten
        B, N2 = self.board_flatten.shape
        dev   = self.device

        # Neighbour colors (computed from board_flatten directly)
        neigh_valid_flatten_b = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)
        neigh_cols            = self._get_neighbor_colors_batch()  # (B,N2,4)

        # Same-color adjacency (ignore empties; respect edges)
        same = (neigh_cols == board_flatten.unsqueeze(2)) \
             & (board_flatten.unsqueeze(2) != -1) \
             & neigh_valid_flatten_b                                              # (B,N2,4)

        # Hook & compress
        parent = torch.arange(N2, dtype=IDX_DTYPE, device=dev).expand(B, N2)      # (B,N2)
        parent = self._hook_and_compress(parent, same)
        roots  = parent                                                           # (B,N2)

        # Count unique liberties per root
        is_lib     = (neigh_cols == -1) & neigh_valid_flatten_b                   # (B,N2,4)
        stone_mask = (board_flatten != -1)                                        # (B,N2)

        libs_per_root = torch.zeros(B * N2, dtype=IDX_DTYPE, device=dev)          # (B*N2,)

        batch_map = torch.arange(B, dtype=IDX_DTYPE, device=dev).view(B, 1, 1)
        roots_exp = roots.unsqueeze(2)                                            # (B,N2,1)
        lib_idx   = self.neigh_index_flatten.view(1, N2, 4)                       # (1,N2,4)
        mask      = is_lib & stone_mask.unsqueeze(2)                               # (B,N2,4)

        #Here K is the number of stone→empty edges across the batch.
        #edges → (r,l), (r,l)
        fb = batch_map.expand_as(mask)[mask]                                       # (K,)
        fr = roots_exp.expand_as(mask)[mask]                                       # (K,)
        fl = lib_idx.expand_as(mask)[mask]                                         # (K,)

        # Deduplicate by (batch, root, liberty_point)
        key_root = fb * N2 + fr
        key_lib  = fb * N2 + fl
        pairs    = torch.stack((key_root, key_lib), dim=1)                         # (K,2)

        sort_key     = pairs[:, 0] * (N2 * B) + pairs[:, 1]
        sorted_idx   = sort_key.argsort()
        pairs_sorted = pairs[sorted_idx]
        uniq         = torch.unique_consecutive(pairs_sorted, dim=0)

        libs_per_root.scatter_add_(0, uniq[:, 0], torch.ones_like(uniq[:, 0], dtype=IDX_DTYPE))
        root_libs = libs_per_root.reshape(B, N2)                                   # (B,N2)

        return roots, root_libs

    # ------------------------------------------------------------------
    # Batched pointer-jumping with periodic convergence check
    # ------------------------------------------------------------------
    def _hook_and_compress(self, parent: torch.Tensor, same: torch.Tensor) -> torch.Tensor:
        B, N2 = parent.shape
        neigh_index_flatten_b = self.neigh_index_flatten.view(1, N2, 4).expand(B, -1, -1)
        max_rounds = N2

        for i in range(max_rounds):
            parent_prev = parent
            nbr_parent  = torch.gather(parent, 1, neigh_index_flatten_b.clamp(min=0).reshape(B, -1)).view(B, N2, 4) # (B,N2,4)
            nbr_parent  = torch.where(same, nbr_parent, torch.full_like(nbr_parent, N2)) # (B,N2,4)
            min_nbr     = nbr_parent.min(dim=2).values                                   # (B,N2)

            hooked = torch.minimum(parent, min_nbr)                                      # (B,N2)
            comp   = torch.gather(hooked, 1, hooked)                                     # parent[parent]
            comp   = torch.gather(comp,   1, comp)                                       # parent[parent[parent]]
            # lazy convergence check
            if (i & 3) == 3 and torch.equal(comp, parent_prev):                          # early exit every 4 iters
                return comp
            parent = comp
        return parent

    # ------------------------------------------------------------------
    # Neighbour helpers (batched, flat graph, 4 dirs)
    # ------------------------------------------------------------------
    def _get_neighbor_colors_batch(
        self,
    ) -> torch.Tensor:
        """Return neighbor colors(stones) pulled from self.board_flatten."""
        B, N2 = self.board_flatten.shape
        # Broadcast per-board tables to batch (views; no copies)
        neigh_index_f_b   = self.neigh_index_flatten.view(1, N2, 4).expand(B, -1, -1).clamp(min=0)  # (B,N2,4)
        neigh_valid_f_b = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)               # (B,N2,4) bool
        
        # 2) Make a 3 Dimensioanl view of the board so ranks match index (no copy)
        board3 = self.board_flatten.unsqueeze(2).expand(-1, -1, 4)                   # (B, N2, 4)
        
        # Gather neighbor colors along the N2 dimension
        gathered = torch.gather(board3, dim=1, index=neigh_index_f_b)                                 # (B,N2,4)

        out = torch.where(neigh_valid_f_b, gathered.to(DTYPE_COLOR),
                      torch.full_like(gathered, SENTINEL_NEIGH_COLOR, dtype=DTYPE_COLOR))
        return out  # (B,N2,4)

    def _get_neighbor_roots_batch(
        self,
        roots: torch.Tensor,                         # (B,N2)
    ) -> torch.Tensor:
        B, N2 = roots.shape
        # Broadcast per-board tables to batch (views; no copies)
        neigh_index_f_b   = self.neigh_index_flatten.view(1, N2, 4).expand(B, -1, -1).clamp(min=0)  # (B,N2,4)
        neigh_valid_f_b = self.neigh_valid_flatten.view(1, N2, 4).expand(B, -1, -1)               # (B,N2,4) bool
        
        roots3  = roots.unsqueeze(2).expand(-1, -1, 4)                                      # (B,N2,4)
        gathered = torch.gather(roots3, dim=1, index=neigh_index_f_b)                                 # (B,N2,4)
        
        out = torch.where(neigh_valid_f_b, gathered,
                      torch.full_like(gathered, SENTINEL_NEIGH_ROOT, dtype=IDX_DTYPE))

        return out  # (B,N2,4)
