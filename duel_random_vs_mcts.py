#!/usr/bin/env python3
# duel_random_vs_mcts.py – random-vs-MCTS using GameStateMachine + debug history

from __future__ import annotations

import os
import time
import signal

from pprint import pprint

print(f"[PID {os.getpid()}] duel_random_vs_mcts start", flush=True)


import torch

from engine.game_state import create_empty_game_state
from engine.game_state_machine import GameStateMachine   # the REAL physics engine
from agents.random_bot import RandomBot
from agents.mcts_bot   import MCTSBot

from utils.game_history import GameHistory
from utils.save_debug_history_minimal import save_debug_games_minimal



from utils.shared import (
    select_device,
    print_timing_report,
    print_performance_metrics,
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



    B_tracked = num_games  # or num_games_to_save if you want

    game_history = GameHistory(
        T_max=max_plies,
        B_tracked=B_tracked,
        H=board_size,
        device=device,
    )

    # Two bots
    bot_A = RandomBot() 
    bot_B = MCTSBot(max_nodes=512, max_depth=256, num_simulations=10)


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


            # --- choose which bot based on ply ---
            if ply < max_plies-1:
                # all earlier moves: pure random
                moves = bot_A.select_moves(real_state_machine)
            else:
                moves = bot_B.select_moves(real_state_machine)

            game_history.moves[:, ply].copy_(moves[:B_tracked])


            real_state_machine.state_transition(moves)


            # record next state
            game_history.boards[:, ply + 1].copy_(real_state_machine.boards[:B_tracked])
            game_history.to_play[:, ply + 1].copy_(real_state_machine.to_play[:B_tracked])
            game_history.hashes[:, ply + 1].copy_(real_state_machine.zobrist_hash[:B_tracked, 0])
        

            ply += 1
                       

    outcome_ratios= real_state_machine.outcome_ratios(komi=komi)
    print("about_to_print_outcome_ratio")
    print(outcome_ratios)
    scores = real_state_machine.compute_scores(komi=komi)       # (B,2)
    print(scores[0:4])
    finished_flags = real_state_machine.is_game_over()          # (B,)
    win_or_loss = real_state_machine.game_outcomes(komi=komi) 
    

    game_history.finalize(
        num_plies=max_plies,
        finished=finished_flags[:B_tracked],
        scores=scores[:B_tracked],
        win_or_loss = win_or_loss[:B_tracked]
    )

    save_debug_games_minimal(
        history=game_history,
        out_dir=history_dir,
        num_games_to_save=num_games_to_save,
    )
    


if __name__ == "__main__":
    simulate_batch_games_with_history(
        num_games=2**6,
        num_games_to_save=2**2,  #to_json
        board_size=9,
        max_plies=100,
        komi=0,
        log_interval=128,
        enable_timing=True,
        history_dir="debug_games",
    )
