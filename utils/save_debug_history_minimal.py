# utils/save_debug_history_minimal.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import torch

from utils.game_history import GameHistory


def save_debug_games_minimal(
    history: GameHistory,
    out_dir: Union[str, Path],
    num_games_to_save: int,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    B = min(num_games_to_save, history.B_tracked)
    T = history.T_actual
    H = history.H

    boards  = history.boards[:B, :T+1].cpu()      # (B, T+1, H, H)
    to_play = history.to_play[:B, :T+1].cpu()     # (B, T+1)
    scores  = history.scores[:B].cpu() if history.scores is not None else None  # (B,2) or None
    

    for g in range(B):
        boards_g  = boards[g].tolist()
        to_play_g = to_play[g].tolist()

        # --- scores ---
        if scores is not None:
            score_black = float(scores[g, 0].item())
            score_white = float(scores[g, 1].item())
        else:
            score_black = None
            score_white = None

        states = []
        for t in range(T + 1):
            states.append(
                {
                    "t": int(t),
                    "to_play": int(to_play_g[t]),  # 0 or 1
                    "board": boards_g[t],          # H×H with {0,1,2}
                }
            )

        payload = {
            "game_index": g,
            "T_actual": T,
            "H": H,
            "W": H,
            "score_black": score_black,
            "score_white": score_white,
            "states": states,
        }

        out_path = out_dir / f"game_{g:03d}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
