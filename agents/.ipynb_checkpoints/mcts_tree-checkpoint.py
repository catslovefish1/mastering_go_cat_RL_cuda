# agents/mcts_tree.py
from __future__ import annotations

from typing import TYPE_CHECKING
import torch
from torch import Tensor

if TYPE_CHECKING:
    from engine.game_state_machine import GameStateMachine


class MCTSTreeIndexInfo:
    """
    Bundled MCTS tree state for a batched Go position.

    Construct directly from a virtual GameStateMachine:

        tree = MCTSTreeIndexInfo(
            engine=search_game_state_machine,
            max_nodes=1024,
            max_depth=256,
        )

    or via the plain helper:

        tree = build_mcts_tree_from_engine_data(
            engine=search_game_state_machine,
            max_nodes=1024,
            max_depth=256,
        )
    """

    def __init__(
        self,
        engine: "GameStateMachine",
        max_nodes: int,    # == max_budget per board
        max_depth: int,    # selection depth limit
    ):
        # -------- infer shape & device from GameStateMachine --------
        boards = engine.boards          # (B,H,W)
        to_play = engine.to_play        # (B,)

        B, H, W = boards.shape
        assert H == W, "Only square boards supported"
        board_size = H
        device = boards.device

        # store scalars
        self.B: int = B
        self.board_size: int = board_size
        self.max_nodes: int = max_nodes
        self.max_depth: int = max_depth
        self.device: torch.device = device

        # derived
        self.M: int = max_nodes
        self.A: int = board_size * board_size + 1  # all points + pass

        # -------- geometry --------
        self.parent: Tensor = torch.full(
            (self.B, self.M), -1, dtype=torch.int64, device=self.device
        )
        self.depth: Tensor = torch.zeros(
            (self.B, self.M), dtype=torch.int16, device=self.device
        )
        self.root_node: Tensor = torch.zeros(
            (self.B,), dtype=torch.int64, device=self.device
        )

        # (row, col) from parent -> this node, (-1,-1) = root/pass
        self.move_pos_from_parent: Tensor = torch.full(
            (self.B, self.M, 2),
            -1,
            dtype=torch.int16,
            device=self.device,
        )

        self.is_expanded: Tensor = torch.zeros(
            (self.B, self.M), dtype=torch.bool, device=self.device
        )
        self.is_terminal: Tensor = torch.zeros(
            (self.B, self.M), dtype=torch.bool, device=self.device
        )
        self.to_play: Tensor = torch.zeros(
            (self.B, self.M), dtype=torch.int8, device=self.device
        )

        # -------- node-level stats --------
        self.N: Tensor = torch.zeros(
            (self.B, self.M), dtype=torch.float32, device=self.device
        )
        self.W: Tensor = torch.zeros(
            (self.B, self.M), dtype=torch.float32, device=self.device
        )
        self.Q: Tensor = torch.zeros(
            (self.B, self.M), dtype=torch.float32, device=self.device
        )

        # per-action prior (policy) and legal mask
        self.P: Tensor = torch.zeros(
            (self.B, self.M, self.A),
            dtype=torch.float16,
            device=self.device,
        )

        # -------- node legal action cache --------
        self.legal: Tensor = torch.zeros(
            (self.B, self.M, self.A),
            dtype=torch.bool,
            device=self.device,
        )

        # -------- root init (node 0 = current position) --------
        self.root_node[:] = 0
        self.parent[:, 0] = -1
        self.depth[:, 0] = 0
        self.move_pos_from_parent[:, 0, :] = -1
        self.is_expanded[:, 0] = False
        self.is_terminal[:, 0] = False
        self.to_play[:, 0] = to_play.to(torch.int8)



def build_mcts_tree_from_engine_data(
    engine: "GameStateMachine",
    max_nodes: int,
    max_depth: int,
) -> MCTSTreeIndexInfo:
    """
    Simple, non-magic helper:
    eat GameStateMachine + params, return a fully-built tree object.
    """
    return MCTSTreeIndexInfo(
        engine=engine,
        max_nodes=max_nodes,
        max_depth=max_depth,
    )

