#!/usr/bin/env python3
# duel_random_physics.py – random-vs-random using GoEnginePhysics + debug history

from __future__ import annotations

import os
import time
import signal

print(f"[PID {os.getpid()}] duel_random_physics start", flush=True)

try:
    import faulthandler
    faulthandler.register(signal.SIGUSR1)
    faulthandler.dump_traceback_later(60, repeat=True)
except Exception as e:
    print(f"[warn] faulthandler setup failed: {e}", flush=True)

import torch

from engine.board_state import create_empty_batch
from engine.board_physics import GoEnginePhysics   # the REAL physics engine
from agents.random_bot import RandomBot

from utils.shared import (
    select_device,
    print_timing_report,
    print_performance_metrics,
)

from utils.board_history_tracker import (
    init_move_history,
    snapshot_pre_move,
    record_move_history,
    save_per_game_histories,
)


def simulate_batch_games_with_history(
    num_games: int = 512,
    board_size: int = 19,
    max_plies: int = 400,
    komi: float = 0,
    log_interval: int = 20,
    enable_timing: bool = True,
    num_games_to_save: int = 5,
    history_dir: str = "debug_games",
):
    device = select_device()
    print(f"Running {num_games} games on {board_size}×{board_size} ({device})")

    # ------------------------------------------------------------------
    # 1) REAL WORLD: create empty batch + real physics engine + bots
    # ------------------------------------------------------------------
    real_state = create_empty_batch(
        batch_size=num_games,
        board_size=board_size,
        device=device,
    )
    # This is the *physical* Go engine that actually mutates boards,
    # pass_count, zobrist_hash, etc.
    real_engine = GoEnginePhysics(real_state)

    # Two bots: later you can swap random_bot_2 → MCTSBot, etc.
    random_bot_1 = RandomBot()   # conceptually "Player 1 / Black"
    random_bot_2 = RandomBot()   # conceptually "Player 2 / White"

    # unify: num_games_to_save drives both tracking and JSON outputs
    num_games_to_save = min(num_games_to_save, num_games)

    # history container for first G games
    move_history = init_move_history(num_games_to_save)

    t0 = time.time()
    ply = 0

    with torch.no_grad():
        while ply < max_plies:
            finished = real_engine.is_game_over()  # (B,) bool on REAL boards
            if finished.all():
                print(f"All games finished by ply {ply}.")
                break

            # --- snapshot BEFORE the move (for history, from REAL engine) ---
            boards_before, to_play_before, hash_before = snapshot_pre_move(
                real_engine, num_games_to_save
            )

            # --- 2) select moves (this calls real_engine.legal_moves()) ---
            # For now, we just alternate bots by ply to make the separation clear.
            # Later: you can still use real_engine.to_play per game if needed.
            if ply % 2 == 0:
                moves = random_bot_1.select_moves(real_engine)  # (B,2) long
            else:
                moves = random_bot_2.select_moves(real_engine)  # (B,2) long

            if ply <= 3:
                print(f"ply {ply} sample moves[0:3]:", moves[:3].cpu().tolist())
                print(f"ply {ply} sample boards for consistency[5:6]:", real_engine.boards[5:6].cpu)

            # --- 3) apply moves to the REAL physics engine ---
            real_engine.state_transition(moves)

            # --- snapshot AFTER the move hash (REAL zobrist) ---
            hash_after = real_engine.zobrist_hash[:, 0].clone()  # (B,)

            # --- 4) record history for first G games ---
            record_move_history(
                move_history=move_history,
                ply=ply,
                moves=moves,
                boards_before=boards_before,
                to_play_before=to_play_before,
                hash_before=hash_before,
                hash_after=hash_after,
                num_games_to_save=num_games_to_save,
            )

            ply += 1

            if log_interval and (ply % log_interval == 0):
                finished = real_engine.is_game_over()
                finished_count = int(finished.sum().item())
                print(f"Ply {ply:4d}: {finished_count}/{num_games} finished")

    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed:.2f}s ({ply} plies simulated)")

    # --- final scoring on the REAL boards ---
    scores = real_engine.compute_scores(komi=komi)       # (B,2) float
    final_hashes = real_engine.zobrist_hash[:, 0]        # (B,) int32
    finished_flags = real_engine.is_game_over()          # (B,) bool

    black_wins = (scores[:, 0] > scores[:, 1]).sum().item()
    white_wins = (scores[:, 1] > scores[:, 0]).sum().item()
    draws = num_games - black_wins - white_wins

    print(f"Black wins: {black_wins} ({black_wins/num_games:.1%})")
    print(f"White wins: {white_wins} ({white_wins/num_games:.1%})")
    print(f"Draws     : {draws} ({draws/num_games:.1%})")

    # --- save per-game JSON histories for first num_games_to_save games ---
    save_per_game_histories(
        base_dir=history_dir,
        move_history=move_history,                       # len = num_games_to_save
        scores=scores[:num_games_to_save],
        final_hashes=final_hashes[:num_games_to_save],
        finished_flags=finished_flags[:num_games_to_save],
        board_size=board_size,
        max_plies=max_plies,
    )

    # --- timing ---
    if enable_timing:
        print_timing_report(real_engine)
        print_timing_report(real_engine.legal_checker)
        print_performance_metrics(elapsed, ply, num_games)

    # --- GPU memory (CUDA only) ---
    if device.type == "cuda":
        MB = 1024 ** 2
        peak_alloc = torch.cuda.max_memory_allocated(device) / MB
        peak_res   = torch.cuda.max_memory_reserved(device) / MB
        cur  = torch.cuda.memory_allocated(device) / MB
        res  = torch.cuda.memory_reserved(device) / MB
        print(f"[CUDA][FINAL] current={cur:.1f} MB reserved={res:.1f} MB")
        print(f"[CUDA][FINAL] peak_alloc={peak_alloc:.1f} MB peak_reserved={peak_res:.1f} MB")


if __name__ == "__main__":
    simulate_batch_games_with_history(
        num_games=2**12,
        board_size=13,
        max_plies=500,
        komi=0,
        log_interval=128,
        enable_timing=True,
        num_games_to_save=4,
        history_dir="debug_games",
    )
