#!/usr/bin/env python3
# duel_random_vs_mcts.py – random-vs-MCTS using GameStateMachine + debug history

from __future__ import annotations

import os
import time
import signal

from pprint import pprint

print(f"[PID {os.getpid()}] duel_random_vs_mcts start", flush=True)

try:
    import faulthandler
    faulthandler.register(signal.SIGUSR1)
    faulthandler.dump_traceback_later(60, repeat=True)
except Exception as e:
    print(f"[warn] faulthandler setup failed: {e}", flush=True)

import torch

from engine.game_state import create_empty_game_state
from engine.game_state_machine import GameStateMachine   # the REAL physics engine
from agents.random_bot import RandomBot
from agents.mcts_bot   import MCTSBot

from utils.game_history import GameHistory


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
    print("Hello World, random_vs_mcts", f"Running {num_games} games on {board_size}×{board_size} ({device})")

    # ------------------------------------------------------------------
    # Intialize game state state = 0
    # ------------------------------------------------------------------

    
    real_state = create_empty_game_state(
        batch_size=num_games,
        board_size=board_size,
        device=device,
    )

    real_state_machine = GameStateMachine(real_state)

    print(
        f"intialized boards[0]:",
        real_state_machine.boards[0].tolist(),
    )

    print(
        f"intialized boards[0]:",
        real_state_machine.boards[0].shape,
    )

    B_tracked = num_games  # or num_games_to_save if you want

    game_history = GameHistory(
        T_max=max_plies,
        B_tracked=B_tracked,
        H=board_size,
        device=device,
    )

    # Two bots
    bot_A = RandomBot()   # conceptually "Player 0 / Black"
    # max_nodes means max_budget here to build MCTS tree
    bot_B = MCTSBot(max_nodes=512, max_depth=256, num_simulations=10)

    # unify: num_games_to_save drives both tracking and JSON outputs
    num_games_to_save = min(num_games_to_save, num_games)

    # history container for first G games
    move_history = init_move_history(num_games_to_save)

    # t = 0 state
    game_history.boards[:,0].copy_(real_state_machine.boards[:B_tracked])
    game_history.to_play[:,0].copy_(real_state_machine.to_play[:B_tracked])
    game_history.hashes[:,0].copy_(real_state_machine.zobrist_hash[:B_tracked, 0])

    # game_history 

    t0 = time.time()
    ply = 0

    LAST_PLY = max_plies - 1  # e.g. 299 if max_plies = 300
    print("LAST_PLY", LAST_PLY)

    with torch.no_grad():
        while ply < max_plies:

            print(f"Ply {ply:4d}: started")

            boards_before, to_play_before, hash_before = snapshot_pre_move(
                real_state_machine, num_games_to_save
            )

            # --- choose which bot based on ply ---
            if ply < max_plies-1:
                # all earlier moves: pure random
                moves = bot_A.select_moves(real_state_machine)
            else:

                # last move: use MCTS
                # print("real_state_last ply_about_to_play")
                # print("real_state_last ply_to_play", real_state_machine.to_play[0:])
                moves = bot_B.select_moves(real_state_machine)

            game_history.moves[:, ply].copy_(moves[:B_tracked])


            # print(f"ply {ply} sample moves[0:]:", moves[0:].cpu().tolist())

            real_state_machine.state_transition(moves)

            print("real_state.boards[0]", real_state.boards[3])


            
            # record next state
            game_history.boards[:, ply + 1].copy_(real_state_machine.boards[:B_tracked])
            game_history.to_play[:, ply + 1].copy_(real_state_machine.to_play[:B_tracked])
            game_history.hashes[:, ply + 1].copy_(real_state_machine.zobrist_hash[:B_tracked, 0])
                

            
            # print("real_state_last ply_to_play", real_state_machine.to_play[0:])




            # print(
            #     f"ply {ply} sample boards for consistency[0]:",
            #     real_state_machine.boards[0].cpu().tolist(),
            # )

            # --- snapshot AFTER the move hash (REAL zobrist) ---
            hash_after = real_state_machine.zobrist_hash[:, 0].clone()  # (B,)



            print(f"Ply {ply:4d}: finised")

            ply += 1
                       
    num_plies = ply

    scores = real_state_machine.compute_scores(komi=komi)       # (B,2)
    finished_flags = real_state_machine.is_game_over()          # (B,)

    game_history.finalize(
        num_plies=num_plies,
        finished=finished_flags[:B_tracked],
        scores=scores[:B_tracked],
    )

    print("Sanity check for game_history")
    pprint(game_history.boards[3].tolist(), width=80, compact=True)


    xxx= game_history.scores[:,0] - game_history.scores[:,1]
    pprint( xxx[0:16].tolist(), width=80, compact=True)

    
    # now game_history is your final instance
    # you can later use game_history.boards, game_history.moves, game_history.T_actual, etc.


    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed:.2f}s ({ply} plies simulated)")

    # --- final scoring on the REAL boards ---
    scores = real_state_machine.compute_scores(komi=komi)       # (B,2) float
    final_hashes = real_state_machine.zobrist_hash[:, 0]        # (B,) int32
    finished_flags = real_state_machine.is_game_over()          # (B,) bool

    # ---- 1) derive wins from scores (old way) ----
    black_wins_scores = (scores[:, 0] > scores[:, 1]).sum().item()
    white_wins_scores = (scores[:, 1] > scores[:, 0]).sum().item()
    draws_scores      = num_games - black_wins_scores - white_wins_scores

    print(f"[scores] Black wins: {black_wins_scores} ({black_wins_scores/num_games:.1%})")
    print(f"[scores] White wins: {white_wins_scores} ({white_wins_scores/num_games:.1%})")
    print(f"[scores] Draws     : {draws_scores} ({draws_scores/num_games:.1%})")

    # ---- 2) game_outcomes sanity (per game: -1/0/+1) ----
    outcomes = real_state_machine.game_outcomes(komi=komi)      # (B,) int8
    print(f"outcomes dtype: {outcomes.dtype}, values in {{-1,0,1}}?")
    print("outcomes unique:", torch.unique(outcomes))

    black_wins_outcomes = int((outcomes == 1).sum().item())
    white_wins_outcomes = int((outcomes == -1).sum().item())
    draws_outcomes      = int((outcomes == 0).sum().item())

    print(f"[outcomes] Black wins: {black_wins_outcomes}")
    print(f"[outcomes] White wins: {white_wins_outcomes}")
    print(f"[outcomes] Draws     : {draws_outcomes}")

    # small B sanity print: first few games
    print("sample scores[0:4]:")
    print(scores[:4].cpu())
    print("sample outcomes[0:16]:")
    print(outcomes[:16].cpu())

    # ---- 3) outcome_ratios sanity (aggregated) ----
    ratios = real_state_machine.outcome_ratios(komi=komi)
    print("[ratios] black={black:.3f}, white={white:.3f}, draw={draw:.3f}, total={total}".format(**ratios))

    # Cross-check: these should match
    assert black_wins_scores  == black_wins_outcomes,  "score vs outcome mismatch (black)"
    assert white_wins_scores  == white_wins_outcomes,  "score vs outcome mismatch (white)"
    assert draws_scores       == draws_outcomes,       "score vs outcome mismatch (draws)"


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
        print_timing_report(real_state_machine)
        print_timing_report(real_state_machine.legal_checker)
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
        num_games=2**6,
        board_size=3,
        max_plies=4,
        komi=0,
        log_interval=128,
        enable_timing=False,
        num_games_to_save=2**4,
        history_dir="debug_games",
    )
