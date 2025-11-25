#!/usr/bin/env python3
# duel_last_move_ab.py – last-move A/B test: random vs MCTS from same positions

from __future__ import annotations

import os
import time
import signal

print(f"[PID {os.getpid()}] duel_last_move_ab start", flush=True)

try:
    import faulthandler
    faulthandler.register(signal.SIGUSR1)
    faulthandler.dump_traceback_later(60, repeat=True)
except Exception as e:
    print(f"[warn] faulthandler setup failed: {e}", flush=True)

import torch

from engine.board_state import create_empty_batch, GoBatchState
from engine.board_physics import GoEnginePhysics
from agents.random_bot import RandomBot
from agents.mcts_bot import MCTSBot, clone_batch_state_from_engine

from utils.shared import (
    select_device,
    print_timing_report,
    print_performance_metrics,
)


def simulate_last_move_ab_test(
    num_games: int = 512,
    board_size: int = 19,
    max_plies: int = 400,
    komi: float = 0.0,
    enable_timing: bool = True,
):
    """
    Run random vs random for (max_plies - 1) plies,
    then do a last-move A/B: random vs MCTS from the SAME root positions.
    """
    device = select_device()
    print(f"Running last-move A/B test on {num_games} games, {board_size}×{board_size} ({device})")

    # warmup plies: exactly one less than max_plies
    warmup_plies = max_plies - 1

    # ---- 1) build initial batch + REAL engine ----
    real_state = create_empty_batch(
        batch_size=num_games,
        board_size=board_size,
        device=device,
    )
    real_engine = GoEnginePhysics(real_state)

    random_bot = RandomBot()
    mcts_bot   = MCTSBot(max_nodes=512, max_depth=256, num_simulations=10)

    t0 = time.time()
    ply = 0

    # ---- 2) warmup: random vs random for warmup_plies ----
    with torch.no_grad():
        while ply < warmup_plies:
            finished = real_engine.is_game_over()
            if finished.all():
                print(f"[warmup] all games finished by ply {ply}")
                break

            moves = random_bot.select_moves(real_engine)
            if ply <= 3:
                print(f"warmup ply {ply} sample moves[0:3]:", moves[:3].cpu().tolist())

            real_engine.state_transition(moves)
            ply += 1

    print(f"[warmup] finished at ply={ply} (target warmup_plies={warmup_plies})")

    # ---- 3) freeze this position, record who is to play ----
    finished_before = real_engine.is_game_over()        # (B,)
    root_to_play = real_engine.to_play.clone()          # (B,) 0=black,1=white

    num_alive = int((~finished_before).sum().item())
    print(f"[A/B] positions alive at last move: {num_alive}/{num_games}")

    if num_alive == 0:
        print("[A/B] no alive games at last move, nothing to test.")
        return

    # ---- 4) branch A: one more RANDOM move then score ----
    state_rand = clone_batch_state_from_engine(real_engine)
    engine_rand = GoEnginePhysics(state_rand)

    moves_rand = random_bot.select_moves(engine_rand)
    engine_rand.state_transition(moves_rand)
    scores_rand = engine_rand.compute_scores(komi=komi)   # (B,2)

    # ---- 5) branch B: one more MCTS move then score ----
    state_mcts = clone_batch_state_from_engine(real_engine)
    engine_mcts = GoEnginePhysics(state_mcts)

    moves_mcts = mcts_bot.select_moves(engine_mcts)
    engine_mcts.state_transition(moves_mcts)
    scores_mcts = engine_mcts.compute_scores(komi=komi)   # (B,2)

    # ---- 6) compare from *root player's* POV, only on alive games ----
    dev = device
    B = num_games

    idx_alive = (~finished_before).nonzero(as_tuple=True)[0]   # (B_alive,)
    B_alive = idx_alive.shape[0]

    rtp = root_to_play[idx_alive].to(torch.long)  # (B_alive,)
    other = 1 - rtp
    arange_alive = torch.arange(B_alive, device=dev)

    # random branch
    s_rand_root  = scores_rand[idx_alive][arange_alive, rtp]
    s_rand_other = scores_rand[idx_alive][arange_alive, other]

    rand_win  = (s_rand_root > s_rand_other)
    rand_lose = (s_rand_root < s_rand_other)
    rand_draw = (s_rand_root == s_rand_other)

    # mcts branch
    s_mcts_root  = scores_mcts[idx_alive][arange_alive, rtp]
    s_mcts_other = scores_mcts[idx_alive][arange_alive, other]

    mcts_win  = (s_mcts_root > s_mcts_other)
    mcts_lose = (s_mcts_root < s_mcts_other)
    mcts_draw = (s_mcts_root == s_mcts_other)

    # ---- 7) print comparison ----
    n_rw = int(rand_win.sum().item())
    n_rl = int(rand_lose.sum().item())
    n_rd = int(rand_draw.sum().item())

    n_mw = int(mcts_win.sum().item())
    n_ml = int(mcts_lose.sum().item())
    n_md = int(mcts_draw.sum().item())

    print("\n[A/B] Last-move comparison FROM ROOT PLAYER POV (alive games only)")
    print(f"  Alive games              : {B_alive}")
    print(f"  Random last move   wins  : {n_rw} ({n_rw / B_alive:.1%})")
    print(f"                      loss : {n_rl} ({n_rl / B_alive:.1%})")
    print(f"                      draw : {n_rd} ({n_rd / B_alive:.1%})")
    print(f"  MCTS last move     wins  : {n_mw} ({n_mw / B_alive:.1%})")
    print(f"                      loss : {n_ml} ({n_ml / B_alive:.1%})")
    print(f"                      draw : {n_md} ({n_md / B_alive:.1%})")

    better = mcts_win & ~rand_win
    worse  = rand_win & ~mcts_win

    n_better = int(better.sum().item())
    n_worse  = int(worse.sum().item())

    print(f"\n[A/B] Per-game improvement:")
    print(f"  Games where MCTS turns non-win → win: {n_better}")
    print(f"  Games where random was win but MCTS isn't: {n_worse}")

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.2f}s  (warmup plies: {ply}, max_plies={max_plies})")

    if enable_timing:
        print_timing_report(real_engine)
        print_performance_metrics(elapsed, ply, num_games)


if __name__ == "__main__":
    simulate_last_move_ab_test(
        num_games=2**10,   # 1024 games
        board_size=19,
        max_plies=500,     # warmup will be 499, last move = 500th
        komi=0.0,
        enable_timing=True,
    )
