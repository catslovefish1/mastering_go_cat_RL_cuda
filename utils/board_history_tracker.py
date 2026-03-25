# utils/board_history_tracker.py
from __future__ import annotations

from typing import List, Dict, Tuple
from pathlib import Path
import os
import json

import torch
from torch import Tensor


def init_move_history(num_games_to_save: int) -> List[List[Dict]]:
    """
    Initialize a nested list to store human-facing move records.

    Returns
    -------
    move_history : list[list[dict]]
        Outer length = num_games_to_save.
        move_history[g] will be a list of per-move dicts for game g.
    """
    return [[] for _ in range(num_games_to_save)]


def snapshot_pre_move(
    engine,
    num_games_to_save: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Take a snapshot of engine state *before* an action is applied.

    Parameters
    ----------
    engine : GoEnginePhysics-like object
        Must expose:
          - engine.boards       : (B,N2) int8
          - engine.to_play      : (B,)    int8
          - engine.zobrist_hash : (B,2)  int32  [:,0] = current

    num_games_to_save : int
        Number of leading games to track in detail.

    Returns
    -------
    boards_before : (G,N2) int8
        Board snapshots for games 0..G-1.

    to_play_before : (B,) int8
        Player-to-move for all games before the move.

    hash_before : (B,) int32
        Current Zobrist hash for all games before the move.
    """
    G = num_games_to_save

    boards_before = engine.boards[:G].clone()        # (G,N2)
    to_play_before = engine.to_play.clone()          # (B,)
    hash_before = engine.zobrist_hash[:, 0].clone()  # (B,)

    return boards_before, to_play_before, hash_before


def record_move_history(
    move_history: List[List[Dict]],
    ply: int,                    # 0-based ply index
    action_ids: Tensor,           # (B,) long -- flat action IDs
    board_size: int,
    boards_before: Tensor,        # (G,N2) int8
    to_play_before: Tensor,       # (B,) int8
    hash_before: Tensor,          # (B,) int32
    hash_after: Tensor,           # (B,) int32
    num_games_to_save: int,
) -> None:
    """
    Append one human-readable move record per tracked game (0..G-1).

    Parameters
    ----------
    action_ids : (B,) long
        Flat action IDs: ``0..N2-1`` = board placement, ``N2`` = pass.
        This is the canonical machine identity for each play.

    board_size : int
        Side length H (H == W).

    Other parameters are unchanged from the previous version.
    """
    G = num_games_to_save
    H = board_size
    N2 = H * H

    for g in range(G):
        a = int(action_ids[g].item())
        is_pass = a >= N2
        r = -1 if is_pass else a // H
        c = -1 if is_pass else a % H

        player = int(to_play_before[g].item())
        player_str = "B" if player == 0 else "W"

        board_flat = boards_before[g].tolist()

        move_record: Dict[str, object] = {
            "move_number": ply + 1,  # 1-based human index
            "action_id": a,
            "row": r,
            "col": c,
            "is_pass": bool(is_pass),
            "player": player,
            "player_str": player_str,
            "zobrist_before": int(hash_before[g].item()),
            "zobrist_after": int(hash_after[g].item()),
            "board_state": board_flat,
        }
        move_history[g].append(move_record)


def save_per_game_histories(
    base_dir: str | os.PathLike,
    move_history: List[List[Dict]],
    scores: Tensor,               # (G,2)
    final_hashes: Tensor,         # (G,)
    finished_flags: Tensor,       # (G,)
    board_size: int,
    max_plies: int,
) -> None:
    """
    Save exactly len(move_history) JSON files, one per tracked game:

      base_dir/game_000.json
      base_dir/game_001.json
      ...

    Each file has the structure:

    {
      "game_id": 0,
      "board_size": 19,
      "total_moves": 400,
      "moves_recorded": 400,
      "truncated": false,
      "final_score": { "black": 171, "white": 136 },
      "final_hash": 1374960918,
      "moves": [
        {
          "move_number": 1,
          "row": 3,
          "col": 16,
          "is_pass": false,
          "player": 0,
          "player_str": "B",
          "zobrist_before": 0,
          "zobrist_after": 123456789,
          "board_state": [ -1, -1, -1, ... ]   # length = board_size^2
        },
        ...
      ]
    }
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    scores_cpu = scores.cpu()
    hashes_cpu = final_hashes.cpu()
    finished_cpu = finished_flags.cpu()

    num_games_to_save = len(move_history)

    for g in range(num_games_to_save):
        moves_g = move_history[g]
        moves_recorded = len(moves_g)

        truncated = (
            moves_recorded >= max_plies and not bool(finished_cpu[g].item())
        )

        game_obj = {
            "game_id": g,
            "board_size": board_size,
            "total_moves": moves_recorded,
            "moves_recorded": moves_recorded,
            "truncated": truncated,
            "final_score": {
                "black": float(scores_cpu[g, 0].item()),
                "white": float(scores_cpu[g, 1].item()),
            },
            "final_hash": int(hashes_cpu[g].item()),
            "moves": moves_g,
        }

        fname = base_dir / f"game_{g:03d}.json"
        with fname.open("w", encoding="utf-8") as f:
            json.dump(game_obj, f, indent=2)

    print(f"[DEBUG] saved {num_games_to_save} game JSONs under {base_dir}")
