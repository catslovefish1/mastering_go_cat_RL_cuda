# board_tensor.py – GPU-optimized Go engine (delta-hash super-ko, no board clones)
# with DEBUG trace hooks for placement/removal (no logic changes)

from __future__ import annotations
from typing import Optional, Dict
from dataclasses import dataclass
from collections import defaultdict

import torch
from torch import Tensor


from contextlib import contextmanager

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
        self.batch_size      = batch_size
        self.board_size      = board_size
        self.history_factor  = history_factor
        self.device          = device or select_device()
        self.enable_timing   = enable_timing
        self.enable_super_ko = enable_super_ko
        self.debug_place_trace = debug_place_trace
        
        
        # Register persistent buffers for repeat computation,for zobrish_hash_history
        # These will be allocated once and reused forever
        self.register_buffer('_hist_masked', None, persistent=False)
        self.register_buffer('_hist_sorted', None, persistent=False)
        self.register_buffer('_sort_indices', None, persistent=False)
        self.register_buffer('_search_idx', None, persistent=False)
        self.register_buffer('_gathered_val', None, persistent=False)
        self.register_buffer('_is_repeat_buffer', None, persistent=False)

        # Timing infra
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
        self.N2 = H * W
       # Do this:
        self._cap_vals = torch.zeros((B, self.N2, 4), dtype=torch.int32, device=self.device)
        self._cap_vals.fill_(0)  # Force write to EVERY element
        # reuse across calls
        self._ws_int32 = torch.zeros((4, B, self.N2), dtype=torch.int32, device=self.device)  
        self._ws_int32.fill_(0)  # Force commit all pages

        # Random values for each position and state: empty(0), black(1), white(2)
        torch.manual_seed(42)  # reproducible profiling
        self.zobrist_table = torch.randint(
            0, 2**31,
            (H, W, 3),
            dtype=torch.int32,
            device=self.device
        )

        # Flattened Zobrist: (N2, 3)
        self.Zpos  = self.zobrist_table.view(self.N2, 3).contiguous()
        self.ZposT = self.Zpos.transpose(0, 1).contiguous()   # (3, N2)
        
    def _mem_tick(self, label: str, min_mb: float = 1.0):
        """Print driver growth since the last tick for this label."""

        cur = torch.mps.driver_allocated_memory() / (1024**2)
        key = f"_mem_tick_prev_{label}"
        prev = getattr(self, key, None)
        setattr(self, key, cur)
        if prev is None:
            return
        delta = cur - prev
        if delta >= min_mb:
            alloc = _mem_alloc_mb(self.device)
            print(f"[MPS][{label}] driver={cur:.1f} MB Δ={delta:+.1f} MB, alloc={alloc:.1f} MB")

        

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
        self.register_buffer("current_player", torch.zeros(B, dtype=torch.int8, device=dev))

        # Pass counter (game ends at 2)
        self.register_buffer("pass_count", torch.zeros(B, dtype=torch.int8, device=dev))

        # -------------------- History tracking --------------------
        max_moves = H * W * self.history_factor

        # Keep board_history only for first up-to-16 boards (debug/printing)
        self._board_hist_track_B = 16
        self.register_buffer(
            "board_history",
            torch.full(
                (self._board_hist_track_B, max_moves, H * W),
                -1, dtype=torch.int8, device=dev
            )
        )

        self.register_buffer("move_count", torch.zeros(B, dtype=torch.int16, device=dev))

        # Super-ko tracking with Zobrist hashing
        if self.enable_super_ko:
            # Current hash for each game
            self.register_buffer("current_hash", torch.zeros(B, dtype=torch.int32, device=dev))
            # History of hashes for super-ko detection
            self.register_buffer("hash_history", torch.zeros((B, max_moves), dtype=torch.int32, device=dev))

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
    @mem_probe("legal_moves") 
    def legal_moves(self) -> BoardTensor:
        if self._last_legal_mask is None:

                legal_mask, info = self._compute_legal_core()

        if self.enable_super_ko:
                legal_mask = self._filter_super_ko_vectorized(legal_mask, info)
        self._last_legal_mask = legal_mask
        self._last_info = info
        return self._last_legal_mask


    @timed_method
    def _compute_legal_core(self):
        """Thin timed wrapper around GoLegalMoveChecker.compute_legal_moves_with_captures."""
        return self.legal_checker.compute_batch_legal_and_info(
            board=self.board,
            current_player=self.current_player,
            return_info=True,
        )



    # ======= Super-ko via CSR (no BxN2xN2 tensors) ============================ #
    # @timed_method
    # @mem_probe("_filter_super_ko_vectorized")
    # def _filter_super_ko_vectorized(self, legal_mask: BoardTensor, info: Dict) -> BoardTensor:
    #     device = self.device
    #     B, H, W = legal_mask.shape
    #     N2 = H * W

    #     # --- Placement delta (EMPTY -> player) ---
    #     player = self.current_player.long()                   # (B,)
    #     z_empty = self.ZposT[0]                               # (N2,)  int32
    #     z_by_color = self.ZposT[1:3]                          # (2,N2) int32
    #     z_place = z_by_color[player]                          # (B,N2) int32
    #     place_delta = z_empty ^ z_place                       # (B,N2) int32

    #     # ---------- CSR (indices must be long) ----------
    #     members = info["stone_global_index"].long()           # (K,)
    #     indptr  = info["stone_global_pointer"].long()         # (R+1,)
    #     gptr    = info["group_global_pointer_per_board"].long()  # (B+1,)
    #     cap_local = info["captured_group_local_index"].long() # (B,N2,4)

    #     R = int(indptr.numel() - 1)
    #     K = int(members.numel())

    #     # ===== Fast path =====
    #     if R == 0 or K == 0:
    #         info["group_xor_remove_delta"] = torch.zeros(0, dtype=torch.int32, device=device)
    #         cap_delta = torch.zeros(B, N2, dtype=torch.int32, device=device)
    #         new_hash = (self.current_hash[:, None] ^ place_delta ^ cap_delta)
    #         repeat_mask = self._repeat_mask_from_history(new_hash, legal_mask)
    #         return legal_mask & ~repeat_mask

    #     # ===== Map group rows to boards; stones to groups and boards =====
    #     groups_per_board = (gptr[1:] - gptr[:-1]).long()      # (B,)
    #     group_board = torch.repeat_interleave(
    #         torch.arange(B, device=device, dtype=torch.long),
    #         groups_per_board
    #     )                                                     # (R,)

    #     group_len = (indptr[1:] - indptr[:-1]).long()         # (R,)
    #     g_of_stone = torch.repeat_interleave(
    #         torch.arange(R, device=device, dtype=torch.long),
    #         group_len
    #     )                                                     # (K,)
    #     stone_board = group_board[g_of_stone]                 # (K,)

    #     # ===== Per-stone removal delta without (B,N2) table =====
    #     opp = (1 - player).long()                             # (B,)
    #     opp_for_stone = opp[stone_board]                      # (K,) in {0,1}
    #     z_opp = z_by_color[opp_for_stone, members]            # (K,)  int32
    #     z_emp = z_empty[members]                              # (K,)  int32
    #     per_stone = (z_opp ^ z_emp).to(torch.int32)           # (K,)  int32

    #     # ===== XOR-reduce per group via padded tree reduction (fast, compact) =====
    #     Lmax = int(group_len.max().item())
    #     pad = torch.zeros(R, Lmax, dtype=torch.int32, device=device)  # (R,Lmax)
    #     idxK = torch.arange(K, device=device, dtype=torch.long)
    #     start = indptr[:-1][g_of_stone]                                # (K,)
    #     pos   = idxK - start                                          # (K,)
    #     pad[g_of_stone, pos] = per_stone

    #     width, acc = Lmax, pad
    #     while width > 1:
    #         half = width // 2
    #         acc = acc[:, :half] ^ acc[:, half:half*2]
    #         if width & 1:
    #             acc = acc ^ acc[:, -1:]
    #         width = acc.size(1)
    #     group_xor = acc.squeeze(1).contiguous()                        # (R,) int32

    #     # ===== Candidate capture: `(B,N2,4)` staging (kept to avoid loops) =====
    #     valid = (cap_local >= 0)                                       # (B,N2,4) bool
    #     base = gptr[:-1].view(B, 1, 1)                                 # (B,1,1)
    #     g_global = (base + cap_local.clamp_min(0)).view(-1)            # (B*N2*4,) long

    #     cap_vals = torch.zeros(B, N2, 4, dtype=torch.int32, device=device)  # (B,N2,4)
    #     flat_valid = valid.view(-1)
    #     cap_vals.view(-1)[flat_valid] = group_xor[g_global[flat_valid]]
    #     cap_delta = cap_vals[..., 0] ^ cap_vals[..., 1] ^ cap_vals[..., 2] ^ cap_vals[..., 3]  # (B,N2)

    #     # ===== Super-ko filter =====
    #     new_hash = (self.current_hash[:, None] ^ place_delta ^ cap_delta)  # (B,N2) int32
    #     repeat_mask = self._repeat_mask_from_history(new_hash, legal_mask)

    #     # Save per-group removal deltas for placement
    #     info["group_xor_remove_delta"] = group_xor

    #     return legal_mask & ~repeat_mask
    
    
    
    @timed_method
    @mem_probe("_filter_super_ko_vectorized")
    def _filter_super_ko_vectorized(self, legal_mask: BoardTensor, info: Dict) -> BoardTensor:
        device = self.device
        B, H, W = legal_mask.shape
        N2 = H * W

        # 0) function entry: seed/print only if we crossed threshold since last time


        # ===== Allocate ONE workspace (likely to hit new buckets early on) =====
        workspace   = self._ws_int32
        place_delta = workspace[0]
        cap_delta   = workspace[1]
        new_hash    = workspace[2]
        temp_work   = workspace[3]

        # 1) after big (B,N2) workspace allocation

        # ---- CSR pieces & stats ----
        members = info["stone_global_index"].long()
        indptr  = info["stone_global_pointer"].long()
        gptr    = info["group_global_pointer_per_board"].long()
        cap_local = info["captured_group_local_index"].long()

        R = int(indptr.numel() - 1)
        K = int(members.numel())
        Lmax = int((indptr[1:] - indptr[:-1]).max().item())
        print(f"[STATS] K={K} R={R} Lmax={Lmax}")

        # 2) after CSR introspection (shapes can steer later temps)

        if R == 0 or K == 0:
            info["group_xor_remove_delta"] = torch.zeros(0, dtype=torch.int32, device=device)
            cap_delta = torch.zeros(B, N2, dtype=torch.int32, device=device)
            new_hash  = (self.current_hash[:, None] ^ place_delta ^ cap_delta)
            
            repeat_mask = self._repeat_mask_from_history(new_hash, legal_mask)
            
            # 3) even early-return may touch new buckets (cap_delta/new_hash)
            return legal_mask & ~repeat_mask

        # ---- map stone -> board, per-stone toggle ----
        player = self.current_player.long()
        z_empty = self.ZposT[0]
        z_by_color = self.ZposT[1:3]

        groups_per_board = (gptr[1:] - gptr[:-1]).long()
        group_board = torch.repeat_interleave(torch.arange(B, device=device, dtype=torch.long), groups_per_board)
        group_len = (indptr[1:] - indptr[:-1]).long()
        g_of_stone = torch.repeat_interleave(torch.arange(R, device=device, dtype=torch.long), group_len)
        stone_board = group_board[g_of_stone]
        opp = (1 - player).long()
        opp_for_stone = opp[stone_board]
        z_opp = z_by_color[opp_for_stone, members]
        z_emp = z_empty[members]
        per_stone = (z_opp ^ z_emp).to(torch.int32)

        # 4) after per-stone vector creation (K-sized bucket)

        # ---- bounded-memory group XOR (tiled) ----
        starts = indptr[:-1]
        lens   = group_len
        group_xor = torch.zeros(R, dtype=torch.int32, device=device)

        R_TILE = 2**10
        W_TILE = 2**4
        offs_w = torch.arange(W_TILE, device=device, dtype=torch.long)

        if int(lens.max().item()) > 0:
            for r0 in range(0, R, R_TILE):
                r1 = min(r0 + R_TILE, R)
                s_seg = starts[r0:r1]
                n_seg = lens[r0:r1]
                if int(n_seg.max().item()) == 0:
                    continue

                max_len = int(n_seg.max().item())
                for p in range(0, max_len, W_TILE):
                    rel   = p + offs_w
                    valid = rel[None, :] < n_seg[:, None]
                    idx   = s_seg[:, None] + rel[None, :]
                    vals  = per_stone[idx.clamp_max(per_stone.numel() - 1)]
                    vals  = torch.where(valid, vals, vals.new_zeros(()))

                    # reduction (alloc-free pattern unchanged here; just measuring growth)
                    acc = vals
                    width = acc.size(1)
                    while width > 1:
                        half = width // 2
                        acc = acc[:, :half] ^ acc[:, half:half*2]
                        if width & 1:
                            acc = acc ^ acc[:, -1:]
                        width = acc.size(1)
                    group_xor[r0:r1] ^= acc.squeeze(1)

        # 5) after group XOR (this is where width/shape churn tends to grow driver)

        # ---- candidate capture (B,N2,4) staging ----
        valid = (cap_local >= 0)
        nnz = int(valid.sum().item())
        print(f"[STATS][cap] nnz_valid={nnz} / total={valid.numel()} ({100.0*nnz/valid.numel():.2f}%)")

        # Estimate how many *elements* we’ll scatter this ply per neighbor (cheap)
        per_k = [int(valid[..., k].sum().item()) for k in range(4)]
        print(f"[STATS][cap] per-neighbor nnz: {per_k} total={sum(per_k)}")

        base  = gptr[:-1].view(B, 1, 1)
        g_global = (base + cap_local.clamp_min(0)).view(-1)

        # ---- Probe: ZERO phase ----
        cap_vals = self._cap_vals
        cap_vals.zero_()  # ADD THIS BACK - needed for correctness!

        # ---- Probe: SCATTER phase (do neighbor-by-neighbor) ----

        for k in range(4):
            mask_k = valid[..., k].view(-1)          # (B*N2,)
            if mask_k.any():
                # pick the k-th neighbor channel for g and for cap_vals
                gk = g_global.view(B, N2, 4)[..., k].view(-1)      # (B*N2,)
                # write only within the k-channel slice of cap_vals
                cap_slice = cap_vals.view(B, N2, 4)[..., k].view(-1)  # (B*N2,)
                cap_slice[mask_k] = group_xor[gk[mask_k]]


        # Reduce to cap_delta (unchanged)
        cap_delta = cap_vals[..., 0] ^ cap_vals[..., 1] ^ cap_vals[..., 2] ^ cap_vals[..., 3]



        # 6) after (B,N2,4) staging — typically the biggest single allocation here

        # ---- finalize hashes + repeat filter ----
        new_hash = (self.current_hash[:, None] ^ place_delta ^ cap_delta)
        self._mem_tick("before:repeat_mask_from_history")
        repeat_mask = self._repeat_mask_from_history(new_hash, legal_mask)
        self._mem_tick("after:_repeat_mask_from_history")
        # 7) after computing new_hash & calling repeat check (may allocate tiles)

        info["group_xor_remove_delta"] = group_xor
        return legal_mask & ~repeat_mask




    # @mem_probe("repeat: fake, quick test, only the shape kept)")
    # def _repeat_mask_from_history(self, new_hash: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    #     # ... original expensive path below ...
    #     B, H, W = legal_mask.shape   # (B, N2) bool
    #     return legal_mask.view(B, H, W)
        


    
    # --------- shared helper: history repeat mask (deduped) ------------------ #
    @mem_probe("repeat: sort/search (B,M)")
    def _repeat_mask_from_history(self, new_hash: Tensor, legal_mask: Tensor) -> Tensor:
        """
        new_hash: (B, N2) int32 candidate future hashes
        legal_mask: (B, H, W) bool
        returns: (B, H, W) bool mask where moves repeat a prior hash
        """
        B, H, W = legal_mask.shape
        M = self.hash_history.shape[1]
        hist = self.hash_history                                   # (B, M) int32
        L = self.move_count.clamp_max(M).long()                     # (B,)

        # # --- debug (no-sync) ---
        # if getattr(self, "debug_repeat_shapes", True):
        #     print(f"[repeat] new_hash   shape={tuple(new_hash.shape)} dtype={new_hash.dtype} device={new_hash.device}")
        #     print(f"[repeat] history    shape={tuple(hist.shape)} dtype={hist.dtype} device={hist.device}")
        #     print(f"[repeat] legal_mask shape={tuple(legal_mask.shape)} dtype={legal_mask.dtype} device={legal_mask.device}")
        #     # (Avoid reductions like .max()/.sum() here to prevent syncs.)

        INT32_MIN = torch.iinfo(torch.int32).min
        ar = torch.arange(M, device=self.device).view(1, -1)        # (1, M)
        valid = (ar < L.view(-1, 1))                                # (B, M) bool

        # Mask invalid columns with a sentinel that cannot equal any real hash
        hist_masked = torch.where(valid, hist, torch.full_like(hist, INT32_MIN))
        hist_sorted, _ = torch.sort(hist_masked, dim=1)             # (B, M)


        # Binary search; if not present, 'val' will be != new_hash
        idx = torch.searchsorted(hist_sorted, new_hash, right=True) - 1  # (B, N2) int64
        idx = idx.clamp_min(0)
        val = hist_sorted.gather(1, idx)                               # (B, N2) int32

        is_repeat_flat = (val == new_hash)                              # (B, N2) bool
        return is_repeat_flat.view(B, H, W)
    
    


    
    @timed_method
    @mem_probe("_place_stones")
    def _place_stones(self, positions: PositionTensor) -> None:
        """
        Vectorized stone placement & capture using CSR.
        Uses precomputed per-group XOR deltas from the last legal() call.
        """
        dev = self.device
        H = W = self.board_size
        N2 = H * W

        # Which entries actually play (skip passes / forced passes)
        mask_play = (positions[:, 0] >= 0) & (positions[:, 1] >= 0)     # (B,)
        b_idx = mask_play.nonzero(as_tuple=True)[0]                     # (M,)
        if b_idx.numel() == 0:
            return

        rows = positions[b_idx, 0].long()
        cols = positions[b_idx, 1].long()
        lin  = rows * W + cols                                          # (M,)
        ply  = self.current_player[b_idx].long()                        # (M,)

        # ---- Cached CSR + per-group deltas from last legal() ----
        info     = self._last_info
        cap_loc  = info["captured_group_local_index"]                   # (B,N2,4) long (−1 if none)
        gptr     = info["group_global_pointer_per_board"]               # (B+1,)   long
        indptr   = info["stone_global_pointer"]                         # (R+1,)   long
        members  = info["stone_global_index"]                           # (K,)     long
        grp_xor  = info["group_xor_remove_delta"].to(torch.int32)       # (R,)     int32

        # ---- Incremental Zobrist hash update: placement + captures ----
        # placement delta at (rows, cols): EMPTY -> ply
        Zpos = self.Zpos                                                # (N2,3) int32
        place_delta = Zpos[lin, 0] ^ Zpos[lin, (ply + 1)]               # (M,)   int32

        # up to 4 captured groups per move; map local->global group id
        g_local4  = cap_loc[b_idx, lin]                                 # (M,4)
        valid4    = (g_local4 >= 0)                                     # (M,4) bool
        g_global4 = gptr[b_idx].unsqueeze(1) + g_local4.clamp_min(0)    # (M,4)

        cap4 = torch.zeros_like(g_global4, dtype=torch.int32)           # (M,4) int32
        if grp_xor.numel() and valid4.any():
            cap4[valid4] = grp_xor[g_global4[valid4]]
        cap_delta = cap4[:, 0] ^ cap4[:, 1] ^ cap4[:, 2] ^ cap4[:, 3]   # (M,)   int32

        self.current_hash[b_idx] ^= (place_delta ^ cap_delta)

        # ---- Apply captures: clear stones in captured groups ----
        # Build one flat list of captured groups (length L <= 4*M)
        flat_valid = valid4.view(-1)
        if flat_valid.any():
            g_list = g_global4.view(-1)[flat_valid]                     # (L,)

            starts = indptr[g_list]                                     # (L,)
            ends   = indptr[g_list + 1]                                 # (L,)
            lens   = (ends - starts)                                    # (L,)

            # Which board each captured group belongs to
            groups_per_move = valid4.sum(1).long()                      # (M,)
            board_of_group  = torch.repeat_interleave(b_idx.long(), groups_per_move)  # (L,)

            S = int(lens.sum().item())
            if S > 0:
                # expand per-group into per-stone membership indices
                g_of_stone   = torch.repeat_interleave(
                    torch.arange(g_list.numel(), device=dev, dtype=torch.long), lens
                )                                                       # (S,)
                start_for_stone = starts[g_of_stone]                    # (S,)
                # position within each group (0..len-1)
                prefix = torch.cumsum(torch.nn.functional.pad(lens, (1, 0)), 0)[:-1]  # (L,)
                pos_in_group = torch.arange(S, device=dev, dtype=torch.long) - prefix[g_of_stone]

                member_idx   = start_for_stone + pos_in_group           # (S,)
                captured_lin = members[member_idx]                      # (S,) linear cell ids 0..N2-1
                board_of_stone = torch.repeat_interleave(board_of_group, lens)        # (S,)

                # Scatter EMPTY in one shot
                flat_all = self.board.view(-1)                          # (B*N2,)
                lin_board_cell = board_of_stone * N2 + captured_lin     # (S,)
                flat_all[lin_board_cell] = torch.tensor(
                    Stone.EMPTY, dtype=flat_all.dtype, device=dev
                )

        # ---- Finally, place the new stones ----
        self.board[b_idx, rows, cols] = ply.to(self.board.dtype)


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
        
        B, H, W = self.batch_size, self.board_size, self.board_size
        max_moves = self.board_history.shape[1] if self.board_history.numel() else 0

        # Flatten board once
        flat = self.board.flatten(1)  # (B, H*W)

        # Per-board move index
        move_idx = self.move_count.long()  # (B,)
        # Valid if within max_moves (debug snapshots)
        if max_moves > 0:
            valid_mask = (move_idx < max_moves)
        else:
            valid_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        # board_history: only write for boards < tracked_B
        T = self._board_hist_track_B
        if T > 0 and max_moves > 0:
            b_all   = torch.arange(B, device=self.device)[valid_mask]
            mv_all  = move_idx[valid_mask]
            track_m = (b_all < T)
            if track_m.any():
                b_trk  = b_all[track_m]
                mv_trk = mv_all[track_m]
                self.board_history[b_trk, mv_trk] = flat[b_trk]

        # hash_history: write for all valid boards
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
        self.pass_count = new_pass_count

        # Apply placement/captures for active, non-pass moves only.
        self._place_stones(safe_positions)

        # Switch to next player
        self.switch_player()
        torch.mps.empty_cache()

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
