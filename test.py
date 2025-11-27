import torch
from engine.game_state import create_empty_game_state
from engine.game_state_machine import GameStateMachine
from engine.stones import Stone

B, N = 2, 5
device = torch.device("cuda")

state = create_empty_game_state(B, N, device)
engine = GameStateMachine(state)

# board 0: 1 black stone
engine.boards[0, 0, 0] = Stone.BLACK
# board 1: 1 white stone
engine.boards[1, 0, 0] = Stone.WHITE

scores = engine.compute_scores(komi=0.5)
print("scores:", scores)          # expect [[1, 0.5], [0, 1.5]]
print("outcomes:", engine.game_outcomes(komi=0.5))
# expect [1, -1] from Black's perspective
