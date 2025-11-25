# agents/mcts_bot.py
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor

from engine.board_state import GoBatchState
from engine.board_physics import GoEnginePhysics
from .mcts_tree import (
    MCTSTreeIndexInfo,
    build_mcts_tree_from_engine_data,
)
from .mcts_ops import run_mcts_random_root


# ---------------------------------------------------------------------
# Helper: clone rich state from REAL engine
# ---------------------------------------------------------------------


def clone_batch_state_from_engine(real_engine: GoEnginePhysics) -> GoBatchState:
    """
    Clone the FULL batch state from the REAL engine so that
    search can mutate it freely without touching the real game.
    """
    return GoBatchState(
        boards=real_engine.boards.clone(),
        to_play=real_engine.to_play.clone(),
        pass_count=real_engine.pass_count.clone(),
        zobrist_hash=real_engine.zobrist_hash.clone(),
    )


def actions_to_rc(actions: Tensor, board_size: int) -> Tensor:
    """
    Convert action indices in [0, board_size^2] to (row, col):

      - 0 .. board_size^2-1 => board points
      - board_size^2        => pass => (-1, -1)

    Returns:
      moves: (B,2) long
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


# ---------------------------------------------------------------------
# MCTSBot: API + data flow, using trivial random-root "MCTS"
# ---------------------------------------------------------------------


@dataclass
class MCTSBot:
    max_nodes: int        # == max tree nodes per board
    max_depth: int        # selection depth limit
    num_simulations: int  # for future use

    @torch.no_grad()
    def select_moves(self, real_engine: GoEnginePhysics) -> Tensor:
        """
        For now:
          1. Clone rich state from real_engine into search_state.
          2. Build search_engine = GoEnginePhysics(search_state).
          3. Build tree = build_mcts_tree_from_engine_data(search_engine,...).
          4. Call run_mcts_random_root(tree, search_engine, num_simulations)
             to get root actions.
          5. Convert actions -> (row, col) moves and return.
        """
        # 1) clone rich state
        search_state = clone_batch_state_from_engine(real_engine)
        search_engine = GoEnginePhysics(search_state)  # virtual search physics

        # 2) build bundled tree from the *search* engine
        tree: MCTSTreeIndexInfo = build_mcts_tree_from_engine_data(
            engine=search_engine,
            max_nodes=self.max_nodes,
            max_depth=self.max_depth,
        )

        # 3) run trivial "MCTS" at root (random policy for now)
        actions = run_mcts_random_root(
            tree=tree,
            engine=search_engine,
            num_simulations=self.num_simulations,
            debug=False,  # <<< only for small num_games

        )  # (B,)

        # 4) convert actions -> (row, col) moves
        moves = actions_to_rc(actions, board_size=tree.board_size)  # (B,2)
        return moves
