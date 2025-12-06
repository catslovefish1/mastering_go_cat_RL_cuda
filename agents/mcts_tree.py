# agents/mcts_tree.py
from __future__ import annotations

from typing import TYPE_CHECKING
import torch
from torch import Tensor

if TYPE_CHECKING:
    from engine.game_state import GameState
    from engine.game_state_machine import GameStateMachine


class MCTSTree:
    """
    Bundled MCTS tree state for a batched Go position.

    Design (local tree per game)
    ----------------------------
    - Index is (b, n):
        b ∈ [0, B) : batch / game index
        n ∈ [0, M) : local node index in that game's tree
    - Node 0 is the initial root for every game b.
    - We preallocate capacity for M nodes per game, and track usage via `next_free[b]`.
    - Action space:
        A = H*W + 1  (all board points + pass)
        action indices a ∈ [0, A)
    """

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    def __init__(
        self,
        batch_size: int,
        board_size: int,
        device: torch.device,
        to_play_root: Tensor,   # (B,) int-like
        max_nodes: int,         # == max_budget per board
        max_depth: int,         # selection depth limit
    ) -> None:
        """
        Low-level constructor: build a tree purely from shape + root player info.

        Normally you won't call this directly. Instead use:
          - MCTSTreeIndexInfo.from_state(root_state, max_nodes, max_depth)
          - or MCTSTreeIndexInfo.from_engine(engine, max_nodes, max_depth)
        """
        B = batch_size
        self.B: int = B
        self.board_size: int = board_size
        self.max_nodes: int = max_nodes
        self.max_depth: int = max_depth
        self.device: torch.device = device

        # -------- derived scalars --------
        self.M: int = max_nodes                     # max local nodes per game
        self.A: int = board_size * board_size + 1   # all points + pass
        self.pass_idx: int = self.A - 1

        # -------- root index per game (for future re-rooting) --------
        self.root_index: Tensor = torch.zeros(
            (self.B,),
            dtype=torch.int32,
            device=self.device,
        )  # (B,)

        # -------- geometry (per-node) --------
        # Parent pointer: -1 for root
        self.parent: Tensor = torch.full(
            (self.B, self.M),
            -1,
            dtype=torch.int32,
            device=self.device,
        )  # (B, M)

        # Action from parent -> this node: -1 for root
        self.parent_action: Tensor = torch.full(
            (self.B, self.M),
            -1,
            dtype=torch.int32,
            device=self.device,
        )  # (B, M)

        # Distance from root (root depth = 0)
        self.depth: Tensor = torch.zeros(
            (self.B, self.M),
            dtype=torch.int16,
            device=self.device,
        )  # (B, M)

        # (row, col) from parent -> this node, (-1,-1) = root/pass
        self.move_pos_from_parent: Tensor = torch.full(
            (self.B, self.M, 2),
            -1,
            dtype=torch.int8,
            device=self.device,
        )  # (B, M, 2)

        # Flags
        self.is_expanded: Tensor = torch.zeros(
            (self.B, self.M),
            dtype=torch.bool,
            device=self.device,
        )  # (B, M)

        self.is_terminal: Tensor = torch.zeros(
            (self.B, self.M),
            dtype=torch.bool,
            device=self.device,
        )  # (B, M)

        # Player to move at node (b, n)
        self.to_play: Tensor = torch.zeros(
            (self.B, self.M),
            dtype=torch.int8,
            device=self.device,
        )  # (B, M)

        # -------- child index table (per-node, per-action) --------
        # child_index[b, n, a] = child node index reached from (b, n) by action a
        # or -1 if child not yet created.
        self.child_index: Tensor = torch.full(
            (self.B, self.M, self.A),
            -1,
            dtype=torch.int32,
            device=self.device,
        )  # (B, M, A)

        # -------- node-level stats (per-node) --------
        self.N: Tensor = torch.zeros(
            (self.B, self.M),
            dtype=torch.float16,
            device=self.device,
        )  # (B, M) visit count

        self.W: Tensor = torch.zeros(
            (self.B, self.M),
            dtype=torch.float16,
            device=self.device,
        )  # (B, M) total value sum

        self.Q: Tensor = torch.zeros(
            (self.B, self.M),
            dtype=torch.float16,
            device=self.device,
        )  # (B, M) mean value

        # -------- per-(node, action) stats --------
        self.P: Tensor = torch.zeros(
            (self.B, self.M, self.A),
            dtype=torch.float16,
            device=self.device,
        )  # (B, M, A) policy prior over actions

        # Node legal action cache
        self.legal: Tensor = torch.zeros(
            (self.B, self.M, self.A),
            dtype=torch.bool,
            device=self.device,
        )  # (B, M, A) legal[b, n, a]

        # -------- simple node allocator (per game) --------
        # For each game b: nodes 0 .. next_free[b]-1 are *in use*.
        # Convention: node 0 is the root, so next_free starts at 1.
        self.next_free: Tensor = torch.ones(
            (self.B,),
            dtype=torch.int32,
            device=self.device,
        )  # (B,)

        # -------- init root node (n = 0) for every game --------
        root_n: int = 0

        self.parent[:, root_n] = -1
        self.parent_action[:, root_n] = -1
        self.depth[:, root_n] = 0
        self.move_pos_from_parent[:, root_n, :] = -1
        self.is_expanded[:, root_n] = False
        self.is_terminal[:, root_n] = False

        # to_play_root: (B,) int-like
        self.to_play[:, root_n] = to_play_root.to(torch.int8)

    # ------------------------------------------------------------------
    # High-level constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_state(
        cls,
        state: "GameState",
        max_nodes: int,
        max_depth: int,
    ) -> "MCTSTree":
        """
        Build a tree from a root GameState snapshot.

        This is the natural entry point for MCTS:

            root_state = real_engine.state.clone()
            tree = MCTSTree.from_state(root_state, max_nodes, max_depth)
        """
        return cls(
            batch_size=state.batch_size,
            board_size=state.board_size,
            device=state.device,
            to_play_root=state.to_play,
            max_nodes=max_nodes,
            max_depth=max_depth,
        )

    @classmethod
    def from_engine(
        cls,
        engine: "GameStateMachine",
        max_nodes: int,
        max_depth: int,
    ) -> "MCTSTreeIndexInfo":
        """
        Convenience: build from a GameStateMachine by using its internal state.
        """
        return cls.from_state(
            state=engine.state,
            max_nodes=max_nodes,
            max_depth=max_depth,
        )
