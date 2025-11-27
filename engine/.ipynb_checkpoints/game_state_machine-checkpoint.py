# engine/game_state_machine.py
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor

from .game_state import GameState
from .stones import Stone
from .go_legal_checker import GoLegalChecker
from utils.shared import timed_method

BoardTensor = Tensor  # (B, H, W)


class GameStateMachine:
    """
    Workspace-owning Go engine.

    - Holds a GoGameState internally as its workspace.
    - Unwraps tensors once in __init__:
        self.boards, self.to_play, self.pass_count, self.zobrist_hash
    - legal_moves() uses internal workspace.
    - state_transition(actions) mutates internal workspace in-place.

    Features:
      - Real legality via GoLegalChecker (captures, suicide rules, etc.).
      - Simple-ko filter: forbid moves whose hash == previous hash only.
      - Zobrist hashing via precomputed candidate hashes.
    """

    # ------------------------------------------------------------------ #
    # # 1. Construction / setup                                          #
    # ------------------------------------------------------------------ #
    def __init__(self, game_state: GameState) -> None:
        # keep a reference to the workspace object
        self.state = game_state

        # unwrap tensors as direct references (no clone)
        self.boards = game_state.boards          # (B, H, W)
        self.to_play = game_state.to_play        # (B,)
        self.pass_count = game_state.pass_count  # (B,)
        # (B, 2): [:,0]=current hash, [:,1]=previous hash
        self.zobrist_hash = game_state.zobrist_hash

        # config
        self.board_size = game_state.board_size  # H = W
        self.board_size_N2 = self.N2 = self.board_size ** 2
        self.batch_size = game_state.batch_size  # B
        self.device = game_state.device

        # Rules engine
        self.legal_checker = GoLegalChecker(
            board_size=self.board_size,
            device=self.device,
        )

        # Zobrist + workspaces for candidate-hash building
        self._init_zobrist_tables()

        # caches for reuse between legal_moves() and state_transition()
        self._latest_legal_mask_raw: Tensor | None = None  # before ko-filter
        self._latest_legal_mask: Tensor | None = None      # after ko-filter
        self._latest_legal_info: Dict | None = None
        self._latest_candidate_hashes: Tensor | None = None

        

    # ------------------------------------------------------------------ #
    # Zobrist init + workspaces                                          #
    # ------------------------------------------------------------------ #
    def _init_zobrist_tables(self) -> None:
        """
        Initialize Zobrist hash tables + linear helpers.

        Zpos:  (N2, 3) int32
        ZposT: (3, N2) int32
          - index 0: EMPTY
          - index 1: BLACK
          - index 2: WHITE

        Also allocates:
          - self._hash_ws : (3, B, N2)   int32  (placement, capture, candidate)
          - self._cap_vals : (B, N2, 4)  int32  (4-neighbor capture staging)
          - self._group_xor: (B*N2,)     int32  (per-group XOR scratch)
        """
        B = self.batch_size
        H = W = self.board_size
        N2= H * W
        dev = self.device

        # Zobrist table for per-intersection states
        g = torch.Generator(device=dev)
        g.manual_seed(42)

        table = torch.randint(
            0,
            2**31,
            (H, W, 3),
            dtype=torch.int32,
            device=dev,
            generator=g,
        )
        Zpos = table.view(N2, 3).contiguous()   # (N2, 3)
        self.Zpos = Zpos
        self.ZposT = Zpos.transpose(0, 1).contiguous()  # (3, N2)

        # workspaces (reused every call)
        self._hash_ws = torch.zeros(
            (3, B, N2), dtype=torch.int32, device=dev
        )
        self._cap_vals = torch.zeros(
            (B, N2, 4), dtype=torch.int32, device=dev
        )
        self._group_xor = torch.zeros(
            B * N2, dtype=torch.int32, device=dev
        )

    # ------------------------------------------------------------------ #
    # Cache helpers                                                      #
    # ------------------------------------------------------------------ #
    def _invalidate_latest(self) -> None:
        """Invalidate caches after any state-changing operation."""
        self._latest_legal_mask_raw = None
        self._latest_legal_mask = None
        self._latest_legal_info = None
        self._latest_candidate_hashes = None


    # ------------------------------------------------------------------ #
    # 2. Public API (what the outside world calls)                       #
    # ------------------------------------------------------------------ #
    
    # ------------------------------------------------------------------ #
    # 2.1 Public: legal moves (real rules + simple-ko)                       #
    # ------------------------------------------------------------------ #
    @timed_method
    @torch.no_grad()
    def legal_moves(self) -> Tensor:
        """
        Compute legal-move mask using the real rules engine + simple-ko.

        Returns
        -------
        legal_mask : (B, H, W) bool
        """
        # 1) rules-based legality + candidate hashes
        legal_mask_raw, info, cand_hash = self._get_or_compute_latest()

        # 2) simple-ko filter (previous hash only)
        legal_mask = self._filter_simple_ko_from_previous(legal_mask_raw, cand_hash)

        # cache ko-filtered mask in case caller wants to inspect it
        self._latest_legal_mask = legal_mask

        return legal_mask

    
    # ------------------------------------------------------------------ #
    # 2.2 Public: state transition (workspace, in-place)                 #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def state_transition(self, actions: Tensor) -> GameState:
        """
        Apply one move per game, in-place on the internal workspace.

        Inputs
        ------
        actions : (B, 2) int
            (row, col) per game; if row<0 or col<0 => pass.

        Workspace (internal)
        --------------------
        self.boards       : (B, H, W) int8
        self.to_play      : (B,) int8 (0=black, 1=white)
        self.pass_count   : (B,) int8
        self.zobrist_hash : (B, 2) int32

        Output
        ------
        Returns the same GameState object (mutated).
        """
        boards = self.boards
        to_play = self.to_play
        pass_count = self.pass_count

        # --- 1) ensure candidate hashes from *pre-move* position ---
        # (reuses cached computations if legal_moves() was just called)
        legal_mask_raw, info, candidate_hashes = self._get_or_compute_latest()
        _ = legal_mask_raw  # not used here, but kept for clarity

        # --- 2) normalize actions / masks ---
        rows, cols, is_pass, finished, play_mask = self._normalize_actions(actions)


        # --- 3) update pass_count ---
        pass_count[:] = torch.where(
            is_pass,
            (pass_count + 1).clamp_max(2),
            torch.zeros_like(pass_count),
        )

        # --- 4) update board: captures + placements ---
        self._update_board_for_placement_and_capture(
            rows=rows,
            cols=cols,
            play_mask=play_mask,
            info=info,
        )

        # --- 5) update zobrist from precomputed candidates ---
        self._update_zobrist_for_placement_and_capture(
            rows=rows,
            cols=cols,
            is_pass=is_pass,
            candidate_hashes=candidate_hashes,
        )

        # --- 6) flip to_play for non-finished games ---
        flipped = to_play ^ 1
        to_play[:] = torch.where(finished, to_play, flipped)


        # --- 7) new state => caches invalid ---
        self._invalidate_latest()

        return self.state  # same object, mutated


    # ------------------------------------------------------------------ #
    #  2.3 Public: Game state                                            #
    # ------------------------------------------------------------------ #
    def is_game_over(self) -> Tensor:
        """Check if any games have ended (2 consecutive passes)."""
        return self.pass_count >= 2

    def compute_scores(self, komi: float = 0) -> Tensor:
        """
        Chinese-style area scoring (stones + surrounded empty territory).
        """
        B, H, W = self.boards.shape          # boards: (B, H, W)
        board_flat = self.boards.view(B, -1) # (B, H*W)
        
        black_stones = (board_flat == Stone.BLACK).sum(dim=1)  # (B,)
        white_stones = (board_flat == Stone.WHITE).sum(dim=1)  # (B,)
        
        territory = self.legal_checker.compute_territory(self.boards)  # (B, 2)
        black_territory = territory[:, 0]  # (B,)
        white_territory = territory[:, 1]  # (B,)
        
        black_area = (black_stones + black_territory).to(torch.float32)  # (B,)
        white_area = (white_stones + white_territory).to(torch.float32)  # (B,)
        
        scores = torch.stack([black_area, white_area], dim=1)  # (B, 2)
        scores[:, 1] += komi
        return scores

    def game_outcomes(self, komi: float = 0.0) -> Tensor:
        """
        Return per-game outcome from Black's perspective.
    
        Returns
        -------
        outcomes : (B,) int8
            +1  -> Black wins
             0  -> Draw (exact tie)
            -1  -> White wins
        """
        scores = self.compute_scores(komi=komi)   # (B, 2)
        black = scores[:, 0]
        white = scores[:, 1]
    
        diff = black - white                      # (B,)
        outcomes = diff.sign().to(torch.int8)     # sign: >0 -> 1, 0 -> 0, <0 -> -1
        return outcomes


    def outcome_ratios(self, komi: float = 0.0) -> dict:
        """
        Compute win/draw ratios over the batch.
    
        Returns
        -------
        {
            "black": float,   # fraction of games Black wins
            "white": float,   # fraction of games White wins
            "draw": float,    # fraction of drawn games
            "total": int,     # total number of games B
        }
        """
        outcomes = self.game_outcomes(komi=komi)    # (B,) in {-1,0,1}
        B = int(outcomes.numel())
    
        black_wins = int((outcomes == 1).sum().item())
        white_wins = int((outcomes == -1).sum().item())
        draws      = int((outcomes == 0).sum().item())
    
        return {
            "black": black_wins / B,
            "white": white_wins / B,
            "draw":  draws      / B,
            "total": B,
        }


    

    # ------------------------------------------------------------------ #
    # 3. Legal pipeline (rules + ko + candidate hashes)                  #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _get_or_compute_latest(self) -> Tuple[Tensor, Dict, Tensor]:
        """
        Ensure latest legality/candidate info exists for the current state.

        Returns
        -------
        legal_mask_raw : (B, H, W) bool   – before ko filter
        info           : dict
        candidate_hash : (B, N2) int32
        """
        if (
            self._latest_legal_mask_raw is None
            or self._latest_legal_info is None
            or self._latest_candidate_hashes is None
        ):
            return self._compute_legal_and_candidates()

        return (
            self._latest_legal_mask_raw,
            self._latest_legal_info,
            self._latest_candidate_hashes,
        )


    @timed_method
    @torch.no_grad()
    def _compute_legal_and_candidates(self) -> Tuple[Tensor, Dict, Tensor]:
        """
        Compute rules-based legality, legal_info, and candidate hashes
        from the *current* workspace state.

        Returns
        -------
        legal_mask_raw : (B, H, W) bool   – before ko filter
        info           : dict             – CSR + capture metadata
        candidate_hash : (B, N2) int32    – Zobrist hashes for placements
        """
        boards = self.boards
        to_play = self.to_play

        move_color = torch.where(
        to_play == 0,
        torch.full_like(to_play, Stone.BLACK, dtype=torch.int8),
        torch.full_like(to_play, Stone.WHITE, dtype=torch.int8),
        )

        # torch.where(condition, A, B):
        # Elementwise: if condition[i] is True → pick A[i], else B[i].
        
        legal_mask_raw, info = self.legal_checker.compute_batch_legal_and_info(
            board=boards,
            move_color=move_color,
        )
        candidate_hashes = self._build_candidate_hashes(legal_mask_raw, info)

        self._latest_legal_mask_raw = legal_mask_raw
        self._latest_legal_info = info
        self._latest_candidate_hashes = candidate_hashes

        return legal_mask_raw, info, candidate_hashes

    # ------------------------------------------------------------------ #
    # Internal: build candidate hashes for ALL placements                #
    # ------------------------------------------------------------------ #
    @timed_method
    @torch.no_grad()
    def _build_candidate_hashes(
        self,
        legal_mask: BoardTensor,
        info: Dict,
    ) -> Tensor:
        """
        Compute Zobrist hash for *every* candidate next state (one per board point).

        Inputs
        ------
        legal_mask : (B, H, W) bool
        info       : dict  – CSR + capture metadata from GoLegalChecker

        Returns
        -------
        candidate_hashes : (B, N2) int32
            Flattened over board positions (row*W + col).

        Side-effects
        ------------
        - Writes per-group XOR deltas into info["group_xor_remove_delta"]
          (kept for potential future use; not needed by this class).
        """
        device = self.device
        B, H, W = legal_mask.shape
        N2 = self.N2

        # Workspaces
        hash_workspace   = self._hash_ws
        placement_delta  = hash_workspace[0].view(B, N2)  # (B, N2) int32
        capture_delta    = hash_workspace[1].view(B, N2)  # (B, N2) int32
        candidate_hashes = hash_workspace[2].view(B, N2)  # (B, N2) int32

        cap_vals            = self._cap_vals       # (B, N2, 4) int32
        group_xor_workspace = self._group_xor      # (B * N2,) int32 – we slice [:R]

        # ---------- 1) placement delta (Zobrist) ----------
        current_player = self.to_play.to(torch.int64)   # (B,)

        z_empty = self.ZposT[0]  # (N2,) int32
        z_black = self.ZposT[1]  # (N2,) int32
        z_white = self.ZposT[2]  # (N2,) int32

        placement_delta.zero_()
        placement_delta.add_(z_empty.view(1, N2))

        black_turn = (current_player == 0)
        white_turn = ~black_turn
        placement_delta[black_turn] ^= z_black.view(1, N2)
        placement_delta[white_turn] ^= z_white.view(1, N2)

        # ---------- 2) CSR pieces ----------
        stone_global_index             = info["stone_global_index"]               # (K,)   int32
        stone_global_pointer           = info["stone_global_pointer"]             # (R+1,) int32
        group_global_pointer_per_board = info["group_global_pointer_per_board"]   # (B+1,) int32
        captured_group_local_index     = info["captured_group_local_index"]       # (B, N2, 4) int32

        R = int(stone_global_pointer.numel() - 1)
        K = int(stone_global_index.numel())

        cur_hash = self.zobrist_hash[:, 0]   # (B,) current hash

        # trivial case: no groups / no stones → only placement delta
        if R == 0 or K == 0:
            info["group_xor_remove_delta"] = group_xor_workspace[:0]

            capture_delta.zero_()
            candidate_hashes.copy_(cur_hash.view(B, 1))
            candidate_hashes ^= placement_delta
            return candidate_hashes

        # ---------- 3) per-stone capture delta via CSR ----------
        z_empty_flat = z_empty                    # (N2,) int32
        z_by_color   = self.ZposT[1:3]            # (2, N2) int32

        groups_per_board      = (group_global_pointer_per_board[1:] - group_global_pointer_per_board[:-1])  # (B,)
        groups_per_board_long = groups_per_board.to(torch.int64)

        group_board_index = torch.repeat_interleave(
            torch.arange(B, device=device, dtype=torch.int64),
            groups_per_board_long,
        )  # (R,)

        stones_per_group      = (stone_global_pointer[1:] - stone_global_pointer[:-1])   # (R,)
        stones_per_group_long = stones_per_group.to(torch.int64)

        group_id_for_stone = torch.repeat_interleave(
            torch.arange(R, device=device, dtype=torch.int64),
            stones_per_group_long,
        )  # (K,)

        stone_board_index   = group_board_index[group_id_for_stone]  # (K,)
        opp_player          = 1 - current_player                     # (B,)
        opp_color_for_stone = opp_player[stone_board_index]          # (K,) ∈ {0,1}

        stone_global_index_long = stone_global_index.to(torch.int64)         # (K,)
        z_opp = z_by_color[opp_color_for_stone, stone_global_index_long]     # (K,)
        z_emp = z_empty_flat[stone_global_index_long]                        # (K,)

        # d_j = Z(opp at stone j) ^ Z(empty at stone j)
        per_stone_delta = (z_opp ^ z_emp)                                    # (K,)

        # ---------- 4) prefix XOR → per-group XOR ----------
        prefix = per_stone_delta.clone()  # (K,)
        offset = 1
        while offset < K:
            prev = prefix.clone()
            prefix[offset:] ^= prev[:-offset]
            offset <<= 1

        start_idx = stone_global_pointer[:-1].to(torch.int64)  # (R,)
        end_idx   = stone_global_pointer[1:].to(torch.int64)   # (R,)

        end_pos   = end_idx - 1
        start_pos = start_idx - 1

        end_val   = prefix[end_pos]
        start_val = torch.zeros_like(end_val)

        non_zero_mask = start_idx > 0
        if non_zero_mask.any():
            start_val[non_zero_mask] = prefix[start_pos[non_zero_mask]]

        group_xor = (end_val ^ start_val)                     # (R,) int32

        group_xor_buf = group_xor_workspace[:R]
        group_xor_buf.copy_(group_xor)

        # ---------- 5) reduce 4-neighbour capture delta per cell ----------
        has_capture = captured_group_local_index >= 0          # (B, N2, 4) bool

        group_offset_per_board = group_global_pointer_per_board[:-1].view(B, 1, 1)  # (B,1,1)

        cap_vals = self._cap_vals  # alias
        cap_vals.copy_(captured_group_local_index.clamp_min(0))
        cap_vals.add_(group_offset_per_board)

        if group_xor_buf.numel() and has_capture.any():
            flat_group_ids = cap_vals[has_capture].to(torch.int64)  # (K',)
            cap_vals.zero_()
            cap_vals[has_capture] = group_xor_buf[flat_group_ids].to(torch.int32)
        else:
            cap_vals.zero_()

        capture_delta.zero_()
        capture_delta.copy_(cap_vals[..., 0])
        capture_delta ^= cap_vals[..., 1]
        capture_delta ^= cap_vals[..., 2]
        capture_delta ^= cap_vals[..., 3]

        # ---------- 6) build candidate hashes ----------
        candidate_hashes.copy_(cur_hash.view(B, 1))
        candidate_hashes ^= placement_delta
        candidate_hashes ^= capture_delta   #(B,M)


        return candidate_hashes             #(B,M)

    # ------------------------------------------------------------------ #
    # Simple-ko filter from previous hash                                #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _filter_simple_ko_from_previous(
        self,
        legal_mask: BoardTensor,
        candidate_hashes: Tensor,  # (B, N2)
    ) -> BoardTensor:
        """
        Simple ko: forbid moves whose candidate hash equals *previous* hash.

        Assumes:
          self.zobrist_hash: (B, 2) int32
            [:, 0] = current hash
            [:, 1] = previous hash
        """
        B, H, W = legal_mask.shape
        prev_hash = self.zobrist_hash[:, 1]        # (B,)
        ko_flat = (candidate_hashes == prev_hash.view(B, 1))  # (B, N2) bool
        ko_mask = ko_flat.view(B, H, W)
        return legal_mask & ~ko_mask

    # ------------------------------------------------------------------ #
    # 4. Action normalization + state updates                            #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Internal: normalize actions / masks                                #
    # ------------------------------------------------------------------ #
    def _normalize_actions(
        self,
        actions: Tensor,  # (B, 2)
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Normalize actions and compute masks based on internal workspace.

        Inputs
        ------
        actions : (B, 2) int  (row, col), negative => pass

        Returns
        -------
        rows      : (B,) long - effective rows (with forced passes)
        cols      : (B,) long - effective cols
        is_pass   : (B,) bool - True where pass
        finished  : (B,) bool - pass_count >= 2 before this move
        play_mask : (B,) bool - True where we actually place stones
        """
        device = self.device
        actions = actions.to(device)

        rows = actions[:, 0]
        cols = actions[:, 1]

        pass_count = self.pass_count  # already on device

        # games already finished (2 passes)
        finished = pass_count >= 2  # (B,) bool

        # finished games are forced to pass (no effect on board)
        eff_rows = torch.where(finished, torch.full_like(rows, -1), rows)
        eff_cols = torch.where(finished, torch.full_like(cols, -1), cols)

        # pass if row<0 or col<0
        is_pass = (eff_rows < 0) | (eff_cols < 0)

        # games where we actually place stones this step
        play_mask = (~is_pass) & (~finished)  # (B,)


        

        return eff_rows.long(), eff_cols.long(), is_pass, finished, play_mask




    # ------------------------------------------------------------------ #
    # Board update: placements + captures                                #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _update_board_for_placement_and_capture(
        self,
        rows: Tensor,         # (B,) long effective rows (may be -1)
        cols: Tensor,         # (B,) long effective cols
        play_mask: Tensor,    # (B,) bool
        info: Dict,
    ) -> None:
        """
        Apply captures (if any) and then place stones on the board for the
        subset of games that actually played (play_mask == True).
        """
        if not play_mask.any():
            return  # no actual placements this ply

        dev = self.device
        B = self.batch_size
        H = W = self.board_size
        N2 = self.N2

        boards = self.boards

        # Subset of boards that actually play
        active_idx = play_mask.nonzero(as_tuple=True)[0]  # (M,)
        r_play = rows[active_idx].to(torch.int64)         # (M,)
        c_play = cols[active_idx].to(torch.int64)         # (M,)
        lin_pos = r_play * W + c_play                     # (M,)

        # Player index: 0 = black to move, 1 = white to move
        player_idx = self.to_play[active_idx]             # (M,) int8

        # Map player index -> board stone encoding (Stone.BLACK / Stone.WHITE).
        # This keeps the board representation independent of the to_play convention
        # and remains correct if Stone uses {EMPTY=0, BLACK=1, WHITE=2}.
        current_color = torch.where(
            player_idx == 0,
            torch.full_like(player_idx, Stone.BLACK, dtype=torch.int8),
            torch.full_like(player_idx, Stone.WHITE, dtype=torch.int8),
        ).to(torch.int64)                                  # (M,)

        # CSR / capture metadata
        captured_group_local_index = info["captured_group_local_index"]          # (B, N2, 4) int32
        group_global_pointer_per_board = info["group_global_pointer_per_board"]  # (B+1,) int32
        stone_global_pointer = info["stone_global_pointer"]                      # (R+1,) int32
        stone_global_index = info["stone_global_index"]                          # (K,)   int32

        # ---- Captured groups for each played move ----
        local_group_ids_4 = captured_group_local_index[active_idx, lin_pos]  # (M, 4)
        valid_slots = (local_group_ids_4 >= 0)                               # (M, 4) bool

        group_offsets = group_global_pointer_per_board[active_idx].unsqueeze(1)  # (M,1)
        global_group_ids_4 = group_offsets + local_group_ids_4.clamp_min(0)     # (M,4) int32

        # ---- Apply captures: clear stones in captured groups ----
        flat_valid = valid_slots.view(-1)
        if flat_valid.any():
            captured_groups_flat = global_group_ids_4.view(-1)[flat_valid]       # (L,) int32

            captured_groups_flat_long = captured_groups_flat.to(torch.int64)
            stone_global_pointer_long = stone_global_pointer.to(torch.int64)

            group_start = stone_global_pointer_long[captured_groups_flat_long]        # (L,) long
            group_end = stone_global_pointer_long[captured_groups_flat_long + 1]      # (L,) long
            stones_per_group = (group_end - group_start)                              # (L,) long

            # Which board each captured group belongs to
            groups_per_move = valid_slots.sum(1).to(torch.int64)                  # (M,)
            board_for_group = torch.repeat_interleave(
                active_idx.to(torch.int64), groups_per_move
            )                                                                    # (L,)

            total_captured = int(stones_per_group.sum())
            if total_captured > 0:
                # Reconstruct stone indices for all captured groups
                group_id_for_stone = torch.repeat_interleave(
                    torch.arange(captured_groups_flat_long.numel(), device=dev, dtype=torch.int64),
                    stones_per_group,
                )                                                                # (S,)
                start_for_stone = group_start[group_id_for_stone]                # (S,)
                prefix_lengths = torch.cumsum(
                    torch.nn.functional.pad(stones_per_group, (1, 0)), 0
                )[:-1]                                                           # (L,)
                pos_in_group = (
                    torch.arange(total_captured, device=dev, dtype=torch.int64)
                    - prefix_lengths[group_id_for_stone]
                )

                stone_index_in_csr = start_for_stone + pos_in_group              # (S,)
                stone_global_index_long = stone_global_index.to(torch.int64)
                captured_lin = stone_global_index_long[stone_index_in_csr]       # (S,)
                board_for_stone = torch.repeat_interleave(
                    board_for_group, stones_per_group
                )                                                                # (S,)

                flat_board = boards.view(-1)                                     # (B * N2,) int8
                lin_board_cell = board_for_stone * N2 + captured_lin             # (S,) long
                flat_board[lin_board_cell] = Stone.EMPTY

        # ---- Finally, place the new stones ----
        boards[active_idx, r_play, c_play] = current_color.to(boards.dtype)

    # ------------------------------------------------------------------ #
    # Zobrist update: placements + captures                             #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _update_zobrist_for_placement_and_capture(
        self,
        rows: Tensor,
        cols: Tensor,
        is_pass: Tensor,
        candidate_hashes: Tensor,
    ) -> None:
        B = self.batch_size
        H = self.board_size
    
        zob = self.zobrist_hash          # (B, 2) int32
        cur = zob[:, 0]                  # (B,) current
        prev = zob[:, 1]                 # (B,) previous
    

    
        # 1) shift ring buffer: prev := old current
        prev[:] = cur
    
        # 2) for non-pass moves, update current from candidate table
        play_mask = ~is_pass             # (B,) bool

        # total = play_mask.numel()                     # total length B
        # num_true = int(play_mask.sum().item())       # True count
        # num_false = total - num_true                 # False count
        
        # print("_update_zobrist_for_placement_and_captur", f"play_mask: total={total}, true={num_true}, false={num_false}")

    
        if play_mask.any():
            idx = play_mask.nonzero(as_tuple=True)[0]   # (M,)
            r = rows[idx]                               # (M,)
            c = cols[idx]                               # (M,)
            lin = r * H + c                             # (M,) long
    
            cur[idx] = candidate_hashes[idx, lin]       # pick new hash for that move


#-----------end of  engine/game_state_machine.py-----------------
