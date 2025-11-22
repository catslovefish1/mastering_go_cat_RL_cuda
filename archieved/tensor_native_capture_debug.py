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
        history_factor: int = 10,
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
        self._last_capture_info = None

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
            0, 2**31,  # int32 range; stored as int64
            (H, W, 3),
            dtype=torch.long,
            device=self.device
        )

        # Linear helpers
        # POS_2D[r,c] -> linear index in [0, N2)
        self.POS_2D = torch.arange(self.N2, device=self.device, dtype=torch.long).view(H, W)
        # Flattened Zobrist: (N2, 3)
        self.Zpos = self.zobrist_table.view(self.N2, 3).contiguous()

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
            torch.zeros(B, dtype=torch.uint8, device=dev)
        )

        # Pass counter (game ends at 2)
        self.register_buffer(
            "pass_count",
            torch.zeros(B, dtype=torch.uint8, device=dev)
        )

        # History tracking
        max_moves = H * W * self.history_factor
        self.register_buffer(
            "board_history",
            torch.full((B, max_moves, H * W), -1, dtype=torch.int8, device=dev)
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
                torch.zeros(B, dtype=torch.long, device=dev)
            )

            # History of hashes for super-ko detection
            self.register_buffer(
                "hash_history",
                torch.zeros((B, max_moves), dtype=torch.long, device=dev)
            )

    # ==================== CORE UTILITIES ==================================== #

    def switch_player(self) -> None:
        """Switch current player and invalidate cached legal moves."""
        self.current_player = self.current_player ^ 1
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Clear cached legal moves and capture info."""
        self._last_legal_mask   = None
        self._last_capture_info = None

    # ---------- DEBUG HELPERS (no logic changes) ----------
    def _sym(self, v: int) -> str:
        return "." if v == Stone.EMPTY else ("X" if v == Stone.BLACK else "O")

    def _pprint_board(self, title: str = "") -> None:
        b = self.board[0].to("cpu").numpy()
        H, W = b.shape
        if title:
            print(f"\n{title}")
        print("     " + " ".join(f"{c:2d}" for c in range(W)))
        print("     " + "--" * W)
        for r in range(H):
            row = " ".join(f"{self._sym(int(b[r,c])):>2}" for c in range(W))
            print(f"{r:2d} | {row}")
        print("\nLegend: X=Black, O=White, .=Empty")

    def _first_coords(self, mask_2d: torch.Tensor, k: int = 24):
        """Return up to k (r,c) coords where mask_2d is True (for printing)."""
        idx = torch.nonzero(mask_2d, as_tuple=False)
        idx = idx[:k]
        return [(int(r.item()), int(c.item())) for r, c in idx]

    def _debug_trace_place(
        self,
        b_idx: torch.Tensor,         # (M,)
        rows: torch.Tensor,          # (M,)
        cols: torch.Tensor,          # (M,)
        ply: torch.Tensor,           # (M,)
        groups_at_moves: torch.Tensor,  # (M,4)
        cap_mask_2d_before_apply: torch.Tensor,  # (M,H,W)
        title: str,
    ) -> None:
        H = W = self.board_size
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        if self.batch_size == 1:
            self._pprint_board("Board BEFORE placement/removal")

        M = b_idx.numel()
        for m in range(M):
            b = int(b_idx[m].item())
            r = int(rows[m].item())
            c = int(cols[m].item())
            p = int(ply[m].item())
            roots_dirs = [int(x) for x in groups_at_moves[m].tolist()]
            cap_cnt = int(cap_mask_2d_before_apply[m].sum().item())
            print(f"\nMove {m}: batch={b} play={'B' if p==0 else 'W'} at ({r},{c})")
            print(f"  capture_groups [N,S,W,E] = {roots_dirs}")
            print(f"  cap_mask true count      = {cap_cnt}")
            if cap_cnt > 0:
                coords = self._first_coords(cap_mask_2d_before_apply[m], k=24)
                print(f"  cap_mask sample coords   = {coords}{' ...' if cap_cnt>24 else ''}")
        print("=" * 80)

    # ==================== LEGAL MOVES ======================================= #

    @timed_method
    def legal_moves(self) -> BoardTensor:
        """
        Compute legal moves (and capture info) and apply super-ko filtering.
        """
        if self._last_legal_mask is None:
            legal_mask, cap_info = self.legal_checker.compute_legal_moves_with_captures(
                board=self.board,
                current_player=self.current_player,
                return_capture_info=True,
            )

            if self.enable_super_ko:
                legal_mask = self._filter_super_ko_vectorized(legal_mask, cap_info)

            self._last_legal_mask   = legal_mask
            self._last_capture_info = cap_info

        return self._last_legal_mask

    # ======= Super-ko via delta-hash (no board clones, no .item() syncs) =======

    def _filter_super_ko_vectorized(self, legal_mask: BoardTensor, cap_info: Dict) -> BoardTensor:
        """
        Super-ko pre-filter using delta Zobrist hashes.
        Operates in linear space; considers all positions as candidates,
        masking non-legal ones. No host sync (.item) and no board cloning.
        """
        if self.move_count.max() == 0:
            return legal_mask

        B, H, W = legal_mask.shape
        N2 = H * W

        legal_mask = legal_mask.clone()
        legal_flat = legal_mask.view(B, N2)                        # (B,N2)

        # Consider all positions as candidates; mask with legality (no .item())
        cand_idx = torch.arange(N2, device=self.device).view(1, -1).expand(B, -1)  # (B,N2)
        cand_mask = legal_flat.bool()                                              # (B,N2)
        if not cand_mask.any():
            return legal_mask

        # Player/opponent per board
        player = self.current_player.long()                        # (B,)
        opp    = 1 - player                                        # (B,)

        # Zobrist contribution for placing the stone at each candidate
        place_xor = self.Zpos[cand_idx, player[:, None]]           # (B,N2)

        # Capture info in linear space
        roots = cap_info["roots"]                                  # (B,N2)
        cap_groups = cap_info["capture_groups"]                    # (B,H,W,4)

        # Map cand_idx → (r,c) to fetch capture groups
        r = cand_idx // W                                          # (B,N2)
        c = cand_idx %  W                                          # (B,N2)
        groups_at = cap_groups[torch.arange(B, device=self.device)[:, None], r, c]  # (B,N2,4)
        valid_any = (groups_at >= 0).any(dim=2, keepdim=True)                         # (B,N2,1)
        groups_clamped = groups_at.clamp_min(0)                                       # (B,N2,4)

        # Build (B,N2,N2) capture mask via explicit equality broadcast
        eq = (roots[:, None, :, None] == groups_clamped[:, :, None, :])               # (B,N2,N2,4)
        cap_mask = eq.any(dim=3)                                                      # (B,N2,N2)
        cap_mask = cap_mask & valid_any.expand(-1, -1, cap_mask.size(2))              # (B,N2,N2)
        cap_mask = cap_mask & cand_mask[:, :, None]                                   # (B,N2,N2)

        # XOR of captured opponent stones per candidate
        ZposT = self.Zpos.transpose(0, 1)                                             # (3,N2)
        Zopp  = ZposT[opp]                                                            # (B,N2)

        sel = torch.where(cap_mask, Zopp[:, None, :],
                          torch.zeros(1, 1, N2, dtype=torch.long, device=self.device))
        # Reduce XOR along last dim → (B,N2)
        def xor_reduce_last_dim(x: Tensor) -> Tensor:
            while x.size(2) > 1:
                n = x.size(2)
                x = torch.bitwise_xor(x[:, :, : n//2], x[:, :, n//2 : (n//2)*2])
                if n % 2:
                    x = torch.bitwise_xor(x, x[:, :, -1:].expand_as(x))
            return x.squeeze(2)
        cap_xor = xor_reduce_last_dim(sel)                                             # (B,N2)

        # Candidate hashes by delta
        new_hash = (self.current_hash[:, None] ^ place_xor) ^ cap_xor                 # (B,N2)

        # Compare against full history (mask by move_count)
        max_moves = self.hash_history.shape[1]
        HIST = self.hash_history                                                      # (B,max_moves)
        hist_mask = torch.arange(max_moves, device=self.device)[None, :] < self.move_count[:, None]  # (B,max_moves)

        matches = (new_hash[:, :, None] == HIST[:, None, :]) & hist_mask[:, None, :]  # (B,N2,max_moves)
        is_repeat_flat = matches.any(dim=2) & cand_mask                               # (B,N2)

        # final mask for readability (keep decoupled from legality)
        repeat_mask = is_repeat_flat.view(B, H, W)                                     # (B,H,W)
        final_mask  = legal_mask & ~repeat_mask
        return final_mask

    # ==================== MOVE EXECUTION ==================================== #

    def _update_hash_incremental(self, b_idx: Tensor, rows: Tensor, cols: Tensor,
                                 ply: Tensor, cap_mask_2d: Tensor) -> None:
        """
        Incrementally update current_hash for rows in b_idx:
        XOR in the placed stone and XOR out captured opponent stones.
        """
        if not self.enable_super_ko or b_idx.numel() == 0:
            return

        M = b_idx.numel()
        H = W = self.board_size
        N2 = H * W

        played_lin = self.POS_2D[rows, cols]                   # (M,)
        opp = (1 - ply).long()                                 # (M,)

        # XOR for placed stones
        place_xor = self.Zpos[played_lin, ply]                 # (M,)

        # XOR over captured opponent stones per row (reduce along N2)
        ZposT = self.Zpos.transpose(0, 1)                      # (3,N2)
        Zopp_rows = ZposT[opp]                                 # (M,N2)

        capM = cap_mask_2d.view(M, N2)                         # (M,N2)
        sel = torch.where(capM, Zopp_rows,
                          torch.zeros_like(Zopp_rows, dtype=torch.long))
        x = sel
        while x.size(1) > 1:
            n = x.size(1)
            x = torch.bitwise_xor(x[:, : n//2], x[:, n//2 : (n//2)*2])
            if n % 2:
                x = torch.bitwise_xor(x, x[:, -1:].expand_as(x))
        cap_xor = x.squeeze(1)                                  # (M,)

        self.current_hash[b_idx] ^= (place_xor ^ cap_xor)

    @timed_method
    def _place_stones(self, positions: PositionTensor) -> None:
        """Vectorized stone placement, capture handling, and incremental hash update (DEBUG-TRACED)."""
        H = W = self.board_size
        B = self.batch_size

        # Detect actual plays (exclude passes)
        mask_play = (positions[:, 0] >= 0) & (positions[:, 1] >= 0)
        if not mask_play.any():
            return

        # Ensure capture info is cached (legal moves also fills it)
        if self._last_capture_info is None:
            self.legal_moves()

        # Indices of rows that played
        b_idx = mask_play.nonzero(as_tuple=True)[0]            # (M,)
        rows = positions[b_idx, 0].long()
        cols = positions[b_idx, 1].long()
        ply  = self.current_player[b_idx].long()               # (M,)
        M    = b_idx.numel()

        # --- Build capture mask in linear space with explicit equality (no (H,W) clones) ---
        roots = self._last_capture_info["roots"]               # (B, N2)
        cap_groups = self._last_capture_info["capture_groups"] # (B,H,W,4)

        groups_at_moves = cap_groups[b_idx, rows, cols]        # (M,4)
        valid_dir = (groups_at_moves >= 0)                    # (M,4)

        roots_selected = roots[b_idx]                          # (M,N2)
        eq = (roots_selected[:, :, None] == groups_at_moves[:, None, :]) & valid_dir[:, None, :]
        
        cap_mask = eq.any(dim=2)                               # (M,N2)
        cap_mask_2d = cap_mask.view(M, H, W)                   # (M,H,W)
        cap_mask_2d[torch.arange(M, device=self.device), rows, cols] = False

        # -------------- DEBUG TRACE (before mutating board) --------------
        if self.debug_place_trace:
            self._debug_trace_place(
                b_idx=b_idx,
                rows=rows,
                cols=cols,
                ply=ply,
                groups_at_moves=groups_at_moves,
                cap_mask_2d_before_apply=cap_mask_2d,
                title="TRACE: place-stones (pre-apply)"
            )

        # --- Incremental hash update BEFORE mutating the board (unchanged) ---
        if self.enable_super_ko:
            self._update_hash_incremental(b_idx, rows, cols, ply, cap_mask_2d)

        # --- Place stones and apply removals on the board (UNCHANGED LOGIC) ---
        self.board[b_idx, rows, cols] = ply.to(self.board.dtype)
        self.board[b_idx] = torch.where(cap_mask_2d, Stone.EMPTY, self.board[b_idx])

        # -------------- DEBUG TRACE (after mutating board) --------------
        if self.debug_place_trace and self.batch_size == 1:
            self._pprint_board("Board AFTER placement/removal")
            white_cnt = int((self.board == Stone.WHITE).sum().item())
            black_cnt = int((self.board == Stone.BLACK).sum().item())
            empty_cnt = int((self.board == Stone.EMPTY).sum().item())
            print(f"Counts AFTER -> white={white_cnt} black={black_cnt} empty={empty_cnt}")

    # ------------------------------------------------------------------ #
    # History                                                            #
    # ------------------------------------------------------------------ #
    @timed_method
    def _update_board_history(self) -> None:
        """
        Record the current position for every live game in the batch.
        """
        B, H, W = self.batch_size, self.board_size, self.board_size
        max_moves = self.board_history.shape[1]

        # Flatten current board state
        flat = self.board.flatten(1)  # (B, H*W)

        # Store in history if not at limit - vectorized operation
        move_idx = self.move_count.long()
        valid_mask = move_idx < max_moves

        if valid_mask.any():
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
        if positions.dim() != 2 or positions.size(1) != 2:
            raise ValueError("positions must be (B, 2)")
        if positions.size(0) != self.batch_size:
            raise ValueError(f"batch size mismatch: expected {self.batch_size}, got {positions.size(0)}")

        # Record history BEFORE the move
        self._update_board_history()
        self.move_count += 1

        # Pass handling
        is_pass = ((positions[:, 0] < 0) | (positions[:, 1] < 0)).to(self.device)
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
