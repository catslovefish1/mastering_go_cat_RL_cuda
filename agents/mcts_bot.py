# agents/mcts_bot.py
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor

from engine.game_state_machine import GameStateMachine
from engine.game_state import GameState
from .mcts_tree import MCTSTree
from .mcts_ops import run_mcts_random_root



# ---------------------------------------------------------------------
# MCTSBot: API + data flow, using trivial random-root "MCTS"
# ---------------------------------------------------------------------


@dataclass
class MCTSBot:
    max_nodes: int        # == max tree nodes per board
    max_depth: int        # selection depth limit
    num_simulations: int  # for future use

    @torch.no_grad()
    def select_moves(self, real_game_state_machine: GameStateMachine) -> Tensor:
        """
        For now:
          1. Clone rich state from real_game_state_machine into search_state.
          2. Build search_game_state_machine = GameStateMachine(search_state).
          3. Build tree = build_mcts_tree_from_engine_data(search_game_state_machine,...).
          4. Call run_mcts_random_root(tree, search_game_state_machine, num_simulations)
             to get root actions.
          5. Convert actions -> (row, col) moves and return.
        """
        # 1) clone rich state

        root_state = real_game_state_machine.state.clone()
        root_game_state_machine = GameStateMachine(root_state)
        
        # 2) build bundled tree from the *search* GameStateMachine
        tree = MCTSTree.from_state(root_state, max_nodes=self.max_nodes, max_depth=self.max_depth)

        

        # 3) run trivial "MCTS" at root (random policy for now)
        actions = run_mcts_random_root(
            tree=tree,
            root_game_state_machine=root_game_state_machine,
            num_simulations=self.num_simulations,
            debug=False,
        )
        print(actions.shape)
        print(actions[:10])

        # 4) convert actions -> (row, col) moves
        moves = actions_to_rc(actions, board_size=tree.board_size)  # (B,2)
        return moves



def actions_to_rc(actions: Tensor, board_size: int) -> Tensor:
    """
    Convert action indices in [0, board_size^2] to (row, col):

      - 0 .. board_size^2-1 => board points
      - board_size^2        => pass => (-1, -1)

    Returns
    -------
    moves : (B,2) long
    """
    B = actions.shape[0]
    dev = actions.device
    moves = torch.full((B, 2), -1, dtype=torch.long, device=dev)

    pass_idx = board_size * board_size
    is_pass = actions == pass_idx

    non_pass_idx = (~is_pass).nonzero(as_tuple=True)[0]
    if non_pass_idx.numel() > 0:
        a_np = actions[non_pass_idx]
        rows = a_np // board_size
        cols = a_np % board_size
        moves[non_pass_idx] = torch.stack([rows, cols], dim=1)

    return moves


