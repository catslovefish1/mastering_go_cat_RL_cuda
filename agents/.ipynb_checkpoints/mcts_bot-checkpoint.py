# agents/mcts_bot.py
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor

from engine.game_state_machine import GameStateMachine
from engine.game_state import GameState
from .mcts_tree import MCTSTree
from .mcts_ops import run_mcts_root, actions_to_moves


# ---------------------------------------------------------------------
# MCTSBot: API + data flow, using generic 4-phase MCTS
# ---------------------------------------------------------------------


@dataclass
class MCTSBot:
    max_nodes: int        # max tree nodes per board
    max_depth: int        # selection depth limit

    @torch.no_grad()
    def select_moves(self, real_game_state_machine: GameStateMachine) -> Tensor:
        """
        For now:
          1. Clone rich state from real_game_state_machine into search_state.
          2. Build search_game_state_machine = GameStateMachine(search_state).
          3. Build tree = MCTSTree.from_state(search_state, max_nodes, max_depth).
          4. Call run_mcts_root(...) which will internally run ~max_nodes simulations.
          5. Convert actions -> (row, col) moves and return.
        """
        # 1) clone rich state
        root_state = real_game_state_machine.state.clone()
        root_game_state_machine = GameStateMachine(root_state)

        # 2) build bundled tree from the *search* GameStateMachine
        tree = MCTSTree.from_state(
            root_state,
            max_nodes=self.max_nodes,
            max_depth=self.max_depth,
        )

        # 3) run MCTS at root
        actions = run_mcts_root(
            tree=tree,
            root_game_state_machine=root_game_state_machine,
            debug=True,
        )
        print(actions.shape)
        print(actions[:10])

        # 4) convert actions -> (row, col) moves
        moves = actions_to_moves(actions, board_size=tree.board_size)  # (B, 2)
        return moves

