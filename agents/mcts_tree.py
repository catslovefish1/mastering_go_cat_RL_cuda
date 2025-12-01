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

    Design (for now)
    ----------------
    - This is a *purely local* search tree:
        index = (b, n)
          b ∈ [0, B) : batch / game index
          n ∈ [0, M) : local node index in that game's tree
    - Node 0 is always the root for every game b.
    - We preallocate capacity for M nodes per game, and track usage via `next_free[b]`.
    """

    def __init__(
        self,
        engine: "GameStateMachine",
        max_nodes: int,    # == max_budget per board
        max_depth: int,    # selection depth limit
    ):
        # -------- infer shape & device from GameStateMachine --------
        boards: Tensor = engine.boards        # shape: (B, H, W) int8
        to_play: Tensor = engine.to_play      # shape: (B,) int8

        B, H, W = boards.shape
        assert H == W, "Only square boards supported"
        board_size: int = H
        device = boards.device

        # -------- store scalars --------
        self.B: int = B
        self.board_size: int = board_size
        self.max_nodes: int = max_nodes     # max nodes per game
        self.max_depth: int = max_depth     # selection depth limit
        self.device: torch.device = device

        # -------- derived scalars --------
        self.M: int = max_nodes             # max local nodes per game
        self.A: int = board_size * board_size + 1  # all points + pass

        # -------- geometry (per-node) --------
        self.parent: Tensor = torch.full(
            (self.B, self.M),
            -1,
            dtype=torch.int64,
            device=self.device,
        )  # shape: (B, M) int64, parent[b, n] = parent node index or -1 for root

        self.depth: Tensor = torch.zeros(
            (self.B, self.M),
            dtype=torch.int16,
            device=self.device,
        )  # shape: (B, M) int16, depth[b, n] = distance from root (root=0)

        # (row, col) from parent -> this node, (-1,-1) = root/pass
        self.move_pos_from_parent: Tensor = torch.full(
            (self.B, self.M, 2),
            -1,
            dtype=torch.int16,
            device=self.device,
        )  # shape: (B, M, 2) int16

        self.is_expanded: Tensor = torch.zeros(
            (self.B, self.M),
            dtype=torch.bool,
            device=self.device,
        )  # shape: (B, M) bool

        self.is_terminal: Tensor = torch.zeros(
            (self.B, self.M),
            dtype=torch.bool,
            device=self.device,
        )  # shape: (B, M) bool

        self.to_play: Tensor = torch.zeros(
            (self.B, self.M),
            dtype=torch.int8,
            device=self.device,
        )  # shape: (B, M) int8, player to move at node (b,n)

        # -------- node-level stats (per-node) --------
        self.N: Tensor = torch.zeros(
            (self.B, self.M),
            dtype=torch.float32,
            device=self.device,
        )  # shape: (B, M) float32, visit count

        self.W: Tensor = torch.zeros(
            (self.B, self.M),
            dtype=torch.float32,
            device=self.device,
        )  # shape: (B, M) float32, total value sum

        self.Q: Tensor = torch.zeros(
            (self.B, self.M),
            dtype=torch.float32,
            device=self.device,
        )  # shape: (B, M) float32, mean value = W / max(N,1)

        # -------- per-(node, action) stats --------
        self.P: Tensor = torch.zeros(
            (self.B, self.M, self.A),
            dtype=torch.float16,
            device=self.device,
        )  # shape: (B, M, A) float16, policy prior for actions

        # Node legal action cache
        self.legal: Tensor = torch.zeros(
            (self.B, self.M, self.A),
            dtype=torch.bool,
            device=self.device,
        )  # shape: (B, M, A) bool, legal[b, n, a]

        # -------- simple node allocator (per game) --------
        # For each game b: nodes 0 .. next_free[b]-1 are *in use*.
        # Convention: node 0 is the root, so next_free starts at 1.
        self.next_free: Tensor = torch.ones(
            (self.B,),
            dtype=torch.int32,
            device=self.device,
        )  # shape: (B,) int32

        # -------- init root node (n = 0) for every game --------
        root_n: int = 0  # scalar

        self.parent[:, root_n] = -1                     # root has no parent
        self.depth[:, root_n] = 0                       # depth 0
        self.move_pos_from_parent[:, root_n, :] = -1    # no move leads to root
        self.is_expanded[:, root_n] = False
        self.is_terminal[:, root_n] = False
        self.to_play[:, root_n] = to_play.to(torch.int8)  # copy from engine root


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
