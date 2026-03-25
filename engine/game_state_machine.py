# engine/game_state_machine.py
from __future__ import annotations

import os
from typing import Tuple

import torch
from torch import Tensor


from .stones import Stone
from .game_state import GameState
from .board_rules_checker import BoardRulesChecker, LegalInfo
from utils.shared import timed_method


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment flag."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class GameStateMachine:
    """
    Workspace-owning Go engine.

    Representation policy
    ---------------------
    Point identity : ``point_id in [0, N2)``
    Action identity: ``action_id in [0, N2]``, where ``N2`` = pass
    Board storage  : ``(B, N2)`` int8

    Public API
    ----------
    :meth:`legal_points`            -- ``(B, N2)`` bool, canonical point legality
    :meth:`state_transition`        -- accepts ``(B,)`` action IDs, canonical

    Action-level semantics (handled by :meth:`state_transition`)
    -----------------------------------------------------------------
    - ``action_id in [0, N2)`` -> board placement
    - ``action_id == N2``      -> pass
    - Already-finished games (``pass_count >= 2``) are forced to no-op.

    Workspace
    ---------
    Holds a GameState internally and unwraps its tensors once in __init__:
    ``self.boards``, ``self.to_play``, ``self.pass_count``, ``self.zobrist_hash``.
    """

    # ------------------------------------------------------------------ #
    # # 1. Construction / setup                                          #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        game_state: GameState,
        alloc_per_call_checker: bool | None = None,
        no_cache_latest: bool | None = None,
    ) -> None:
        # keep a reference to the workspace object
        self.state = game_state

        self.alloc_per_call_checker = (
            _env_flag("GO_ENGINE_ALLOC_PER_CALL_CHECKER", False)
            if alloc_per_call_checker is None
            else bool(alloc_per_call_checker)
        )
        self.no_cache_latest = (
            _env_flag("GO_ENGINE_NO_CACHE_LATEST", False)
            if no_cache_latest is None
            else bool(no_cache_latest)
        )

        # unwrap tensors as direct references (no clone)
        self.boards = game_state.boards          # (B, N2)
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
        self.legal_checker = BoardRulesChecker(
            board_size=self.board_size,
            device=self.device,
            alloc_per_call_checker=self.alloc_per_call_checker,
        )

        # Zobrist tables for candidate-hash building
        self._init_zobrist_tables()

        # Caches -- valid for the current board state until the state changes.
        self._latest_legal_points_raw: Tensor | None = None   # (B, N2) before ko filter
        self._latest_legal_points: Tensor | None = None       # (B, N2) after ko filter
        self._latest_legal_info: LegalInfo | None = None
        self._latest_candidate_hashes: Tensor | None = None

    # ------------------------------------------------------------------ #
    # Zobrist init                                                       #
    # ------------------------------------------------------------------ #
    def _init_zobrist_tables(self) -> None:
        """
        Initialize Zobrist hash tables + linear helpers.

        Zpos:  (N2, 3) int32
        ZposT: (3, N2) int32
          - index 0: EMPTY
          - index 1: BLACK
          - index 2: WHITE
        """
        N2 = self.N2
        dev = self.device

        # Zobrist table for per-intersection states
        g = torch.Generator(device=dev)
        g.manual_seed(42)

        Zpos = torch.randint(
            0,
            2**31,
            (N2, 3),
            dtype=torch.int32,
            device=dev,
            generator=g,
        )
        self.Zpos = Zpos
        self.ZposT = Zpos.transpose(0, 1).contiguous()  # (3, N2)

    # ------------------------------------------------------------------ #
    # Cache helpers                                                      #
    # ------------------------------------------------------------------ #
    def _invalidate_latest(self) -> None:
        """Invalidate caches after any state-changing operation."""
        self._latest_legal_points_raw = None
        self._latest_legal_points = None
        self._latest_legal_info = None
        self._latest_candidate_hashes = None


    # ------------------------------------------------------------------ #
    # 2. Public API (what the outside world calls)                       #
    # ------------------------------------------------------------------ #
    
    # ------------------------------------------------------------------ #
    # 2.1 Public: legal points                                           #
    # ------------------------------------------------------------------ #
    @timed_method
    @torch.no_grad()
    def legal_points(self) -> Tensor:
        """
        Returns
        -------
        legal_points : (B, N2) bool
            True where placing a stone is legal (captures, suicide, ko).
            Does not include pass -- pass is always legal at the action level.
        """
        legal_points_raw, _, candidate_hashes = self._get_or_compute_latest()
        legal_points = self._filter_simple_ko(legal_points_raw, candidate_hashes)
        self._latest_legal_points = legal_points
        return legal_points


    # ------------------------------------------------------------------ #
    # 2.2 Public: state transition (workspace, in-place)                 #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def state_transition(self, action_ids: Tensor) -> GameState:
        """
        Parameters
        ----------
        action_ids : (B,) int
            ``0 .. N2-1`` = board placement, ``N2`` = pass.
            Finished games (``pass_count >= 2``) are forced to no-op.

        Returns the same GameState object (mutated in-place).
        """
        to_play = self.to_play
        pass_count = self.pass_count

        _, legal_info, candidate_hashes = self._get_or_compute_latest()

        point_ids, is_pass, finished, play_mask = self._normalize_action_ids(action_ids)

        pass_count[:] = torch.where(
            is_pass,
            (pass_count + 1).clamp_max(2),
            torch.zeros_like(pass_count),
        )

        self._apply_placement_and_capture(point_ids, play_mask, legal_info)
        self._apply_zobrist_update(point_ids, is_pass, candidate_hashes)

        flipped = to_play ^ 1
        to_play[:] = torch.where(finished, to_play, flipped)

        self._invalidate_latest()
        return self.state

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
        black_stones = (self.boards == Stone.BLACK).sum(dim=1)  # (B,)
        white_stones = (self.boards == Stone.WHITE).sum(dim=1)  # (B,)

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
    def _get_or_compute_latest(self) -> Tuple[Tensor, LegalInfo, Tensor]:
        """
        Ensure latest legality/candidate metadata exists for the current state.

        Returns
        -------
        legal_points_raw   : (B, N2) bool   -- before ko filter
        legal_info         : LegalInfo
        candidate_hashes   : (B, N2) int32
        """
        if self.no_cache_latest:
            return self._compute_legal_and_candidates(cache_results=False)

        if (
            self._latest_legal_points_raw is None
            or self._latest_legal_info is None
            or self._latest_candidate_hashes is None
        ):
            return self._compute_legal_and_candidates()

        return (
            self._latest_legal_points_raw,
            self._latest_legal_info,
            self._latest_candidate_hashes,
        )


    @timed_method
    @torch.no_grad()
    def _compute_legal_and_candidates(
        self,
        cache_results: bool = True,
    ) -> Tuple[Tensor, LegalInfo, Tensor]:
        """
        Compute board-point legality, capture metadata, and candidate hashes
        from the *current* workspace state.

        Returns
        -------
        legal_points_raw   : (B, N2) bool   -- before ko filter
        legal_info         : LegalInfo       -- CSR + capture metadata
        candidate_hashes   : (B, N2) int32   -- Zobrist hashes per placement
        """
        boards = self.boards
        to_play = self.to_play

        to_play_color = torch.where(
            to_play == 0,
            torch.full_like(to_play, Stone.BLACK, dtype=torch.int8),
            torch.full_like(to_play, Stone.WHITE, dtype=torch.int8),
        )
        legal_points_raw, legal_info = self.legal_checker.compute_batch_legal_and_info(
            board=boards,
            to_play_color=to_play_color,
        )
        candidate_hashes = self._build_candidate_hashes(legal_info)

        if cache_results:
            self._latest_legal_points_raw = legal_points_raw
            self._latest_legal_info = legal_info
            self._latest_candidate_hashes = candidate_hashes

        return legal_points_raw, legal_info, candidate_hashes

    # ------------------------------------------------------------------ #
    # Internal: build candidate hashes for ALL placements                #
    # ------------------------------------------------------------------ #
    @timed_method
    @torch.no_grad()
    def _build_candidate_hashes(self, legal_info: LegalInfo) -> Tensor:
        """
        Compute Zobrist hash for *every* candidate placement (one per point).

        Pipeline
        --------
        1. Placement delta -- XOR for changing empty->stone at each point.
        2. Per-group capture XOR -- prefix-XOR scan over the CSR stone array
           produces one XOR delta per batch-global group.
        3. Local->global mapping -- board-local capture ids from the checker are
           lifted into batch-global ids via ``group_offset_by_board``,
           then looked up in the per-group XOR table.
        4. Per-point capture delta -- XOR the (up to 4) neighbor capture deltas.
        5. Final hash -- ``current_hash ^ placement_delta ^ capture_delta``.

        Returns
        -------
        candidate_hashes : (B, N2) int32
        """
        device = self.device
        B = self.batch_size
        N2 = self.N2

        # Per-call scratch buffers keep this path simple and stateless.
        placement_delta = torch.zeros((B, N2), dtype=torch.int32, device=device)
        capture_delta = torch.zeros((B, N2), dtype=torch.int32, device=device)
        candidate_hashes = torch.zeros((B, N2), dtype=torch.int32, device=device)
        cap_vals = torch.zeros((B, N2, 4), dtype=torch.int32, device=device)
        group_xor_workspace = torch.zeros(B * N2, dtype=torch.int32, device=device)

        # ---------- Phase 1: placement delta (Zobrist) ----------
        # XOR of Z(empty) with Z(new color) for every point.
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

        # ---------- Phase 2: unpack CSR + capture metadata ----------
        csr = legal_info.csr
        stone_point_ids = csr.stone_point_ids                     # (K,)   int32
        stone_pointer_by_group = csr.stone_pointer_by_group       # (R+1,) int32
        group_offset_by_board = csr.group_offset_by_board         # (B+1,) int32
        captured_group_local_ids = legal_info.captured_group_local_ids  # (B, N2, 4) int32

        R = int(stone_pointer_by_group.numel() - 1)
        K = int(stone_point_ids.numel())

        cur_hash = self.zobrist_hash[:, 0]   # (B,) current hash

        # trivial case: no groups / no stones → only placement delta
        if R == 0 or K == 0:
            capture_delta.zero_()
            candidate_hashes.copy_(cur_hash.view(B, 1))
            candidate_hashes ^= placement_delta
            return candidate_hashes

        # ---------- Phase 3: per-stone Zobrist delta via CSR ----------
        # For each stone in the batch, compute d_j = Z(opp) ^ Z(empty).
        z_by_color = self.ZposT[1:3]               # (2, N2) int32

        groups_per_board = (group_offset_by_board[1:] - group_offset_by_board[:-1])  # (B,)
        groups_per_board_long = groups_per_board.to(torch.int64)

        board_index_per_global_group = torch.repeat_interleave(
            torch.arange(B, device=device, dtype=torch.int64),
            groups_per_board_long,
        )  # (R,)

        stones_per_group = (stone_pointer_by_group[1:] - stone_pointer_by_group[:-1])   # (R,)
        stones_per_group_long = stones_per_group.to(torch.int64)

        global_group_index_for_stone = torch.repeat_interleave(
            torch.arange(R, device=device, dtype=torch.int64),
            stones_per_group_long,
        )  # (K,)

        stone_board_index   = board_index_per_global_group[global_group_index_for_stone]  # (K,)
        opp_player          = 1 - current_player                     # (B,)
        opp_color_for_stone = opp_player[stone_board_index]          # (K,) ∈ {0,1}

        stone_point_ids_long = stone_point_ids.to(torch.int64)         # (K,)
        z_opp = z_by_color[opp_color_for_stone, stone_point_ids_long]  # (K,)
        z_emp = z_empty[stone_point_ids_long]                              # (K,)

        # d_j = Z(opp at stone j) ^ Z(empty at stone j)
        per_stone_delta = (z_opp ^ z_emp)                                    # (K,)

        # ---------- Phase 4: prefix XOR → per-group capture XOR ----------
        # Parallel prefix scan turns per-stone deltas into per-group XOR sums.
        prefix = per_stone_delta.clone()  # (K,)
        offset = 1
        while offset < K:
            prev = prefix.clone()
            prefix[offset:] ^= prev[:-offset]
            offset <<= 1

        start_idx = stone_pointer_by_group[:-1].to(torch.int64)  # (R,)
        end_idx = stone_pointer_by_group[1:].to(torch.int64)     # (R,)

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

        # ---------- 5) local→global group ids → per-point capture XOR delta ----
        # captured_group_local_ids uses -1 as NO_CAPTURE sentinel.
        # Replace sentinels with 0 for safe arithmetic; has_capture masks them out.
        has_capture = captured_group_local_ids >= 0          # (B, N2, 4) bool

        group_offset_per_board = group_offset_by_board[:-1].view(B, 1, 1)

        safe_local_ids = captured_group_local_ids.clamp_min(0)
        cap_vals.copy_(safe_local_ids)
        cap_vals.add_(group_offset_per_board)                  # now batch-global group ids

        if group_xor_buf.numel() and has_capture.any():
            global_group_gather_idx = cap_vals[has_capture].to(torch.int64)
            cap_vals.zero_()
            cap_vals[has_capture] = group_xor_buf[global_group_gather_idx].to(torch.int32)
        else:
            cap_vals.zero_()

        capture_delta.zero_()
        capture_delta.copy_(cap_vals[..., 0])
        capture_delta ^= cap_vals[..., 1]
        capture_delta ^= cap_vals[..., 2]
        capture_delta ^= cap_vals[..., 3]

        # ---------- Phase 6: assemble candidate hashes ----------
        candidate_hashes.copy_(cur_hash.view(B, 1))
        candidate_hashes ^= placement_delta
        candidate_hashes ^= capture_delta

        return candidate_hashes             # (B, N2)

    # ------------------------------------------------------------------ #
    # Simple-ko filter                                                   #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _filter_simple_ko(
        self,
        legal_points: Tensor,       # (B, N2) bool
        candidate_hashes: Tensor,   # (B, N2) int32
    ) -> Tensor:
        """
        Simple ko: forbid placements whose candidate hash equals the
        *previous* board hash.  Returns (B, N2) bool.
        """
        B = legal_points.shape[0]
        prev_hash = self.zobrist_hash[:, 1]                     # (B,)
        ko_mask = (candidate_hashes == prev_hash.view(B, 1))    # (B, N2)
        return legal_points & ~ko_mask

    # ------------------------------------------------------------------ #
    # 4. Action normalization                                            #
    # ------------------------------------------------------------------ #
    def _normalize_action_ids(
        self,
        action_ids: Tensor,  # (B,)
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Classify each action as placement, pass, or terminal no-op.

        Parameters
        ----------
        action_ids : (B,) int
            ``0..N2-1`` = board placement, ``N2`` = pass.

        Returns
        -------
        point_ids : (B,) long -- point index (clamped safe for indexing)
        is_pass   : (B,) bool
        finished  : (B,) bool -- pass_count >= 2
        play_mask : (B,) bool -- actually placing a stone
        """
        N2 = self.N2
        action_ids = action_ids.to(self.device).long()

        finished = self.pass_count >= 2
        is_pass = (action_ids >= N2) | finished
        play_mask = ~is_pass

        point_ids = action_ids.clamp(max=N2 - 1)

        return point_ids, is_pass, finished, play_mask




    # ------------------------------------------------------------------ #
    # Board update: placements + captures                                #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _apply_placement_and_capture(
        self,
        point_ids: Tensor,    # (B,) long -- point index
        play_mask: Tensor,    # (B,) bool
        legal_info: LegalInfo,
    ) -> None:
        """
        Apply captures (if any) then place stones for games where
        ``play_mask`` is True.
        """
        if not play_mask.any():
            return

        dev = self.device
        B = self.batch_size
        N2 = self.N2
        boards = self.boards

        active_idx = play_mask.nonzero(as_tuple=True)[0]           # (M,)
        active_point_ids = point_ids[active_idx].to(torch.int64)   # (M,)

        player_idx = self.to_play[active_idx]                      # (M,) int8
        current_color = torch.where(
            player_idx == 0,
            torch.full_like(player_idx, Stone.BLACK, dtype=torch.int8),
            torch.full_like(player_idx, Stone.WHITE, dtype=torch.int8),
        )                                                          # (M,) int8

        # CSR / capture metadata
        csr = legal_info.csr
        captured_group_local_ids = legal_info.captured_group_local_ids      # (B, N2, 4) int32
        group_offset_by_board = csr.group_offset_by_board                    # (B+1,) int32
        stone_pointer_by_group = csr.stone_pointer_by_group                  # (R+1,) int32
        stone_point_ids = csr.stone_point_ids                                # (K,)   int32

        captured_group_local_ids_active = captured_group_local_ids[active_idx, active_point_ids]  # (M, 4)
        valid_capture_slots = (captured_group_local_ids_active >= 0)                             # (M, 4)

        group_offset_active = group_offset_by_board[active_idx].unsqueeze(1)  # (M, 1)
        safe_local_ids = captured_group_local_ids_active.clamp_min(0)
        captured_global_group_ids = group_offset_active + safe_local_ids

        # ---- Apply captures: clear stones in captured groups ----
        valid_slots_packed = valid_capture_slots.view(-1)
        if valid_slots_packed.any():
            captured_global_ids = captured_global_group_ids.view(-1)[valid_slots_packed]

            captured_global_ids_long = captured_global_ids.to(torch.int64)
            stone_pointer_by_group_long = stone_pointer_by_group.to(torch.int64)

            group_start = stone_pointer_by_group_long[captured_global_ids_long]
            group_end = stone_pointer_by_group_long[captured_global_ids_long + 1]
            stones_per_group = (group_end - group_start)

            groups_per_move = valid_capture_slots.sum(1).to(torch.int64)
            board_for_group = torch.repeat_interleave(
                active_idx.to(torch.int64), groups_per_move
            )

            total_captured = int(stones_per_group.sum())
            if total_captured > 0:
                group_id_for_stone = torch.repeat_interleave(
                    torch.arange(captured_global_ids_long.numel(), device=dev, dtype=torch.int64),
                    stones_per_group,
                )
                start_for_stone = group_start[group_id_for_stone]
                prefix_lengths = torch.cumsum(
                    torch.nn.functional.pad(stones_per_group, (1, 0)), 0
                )[:-1]
                pos_in_group = (
                    torch.arange(total_captured, device=dev, dtype=torch.int64)
                    - prefix_lengths[group_id_for_stone]
                )

                stone_index_in_csr = start_for_stone + pos_in_group
                stone_point_ids_long = stone_point_ids.to(torch.int64)
                captured_point_ids = stone_point_ids_long[stone_index_in_csr]
                board_for_stone = torch.repeat_interleave(
                    board_for_group, stones_per_group
                )

                boards_linear = boards.view(-1)
                captured_linear_idx = board_for_stone * N2 + captured_point_ids
                boards_linear[captured_linear_idx] = Stone.EMPTY

        # ---- Place new stones ----
        boards_linear = boards.view(-1)
        placement_linear_idx = active_idx.to(torch.int64) * N2 + active_point_ids
        boards_linear[placement_linear_idx] = current_color

    # ------------------------------------------------------------------ #
    # Zobrist update                                                     #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _apply_zobrist_update(
        self,
        point_ids: Tensor,         # (B,) long -- point index
        is_pass: Tensor,           # (B,) bool
        candidate_hashes: Tensor,  # (B, N2) int32
    ) -> None:
        zob = self.zobrist_hash          # (B, 2) int32
        cur = zob[:, 0]                  # (B,) current
        prev = zob[:, 1]                 # (B,) previous

        prev[:] = cur

        play_mask = ~is_pass
        if play_mask.any():
            idx = play_mask.nonzero(as_tuple=True)[0]
            cur[idx] = candidate_hashes[idx, point_ids[idx]]
