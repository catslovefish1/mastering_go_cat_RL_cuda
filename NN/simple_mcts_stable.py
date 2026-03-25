"""NN/simple_mcts.py - Simple Monte Carlo Tree Search with random rollouts.

This module implements a minimal MCTS that:
- Uses only random rollouts for evaluation
- Works with TensorBoard for game mechanics
- Maintains simplicity and clarity
- Uses flat tensor-based tree structure (like a hotel with room numbers)
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import math

import torch
from torch import Tensor

# Import engine and utilities
from engine.tensor_native import TensorBoard
from utils.shared import (
    select_device,
    flat_to_2d,
    coords_to_flat,
    create_pass_positions,
    sample_from_mask
)

# ========================= DATA STRUCTURES =========================

@dataclass
class MCTSNode:
    """Single MCTS node representing a board state.
    
    Kept for backward compatibility, but internally we use tensor storage.
    """
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[Tuple[int, int], MCTSNode] = None
    is_expanded: bool = False
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
    
    @property
    def value(self) -> float:
        """Average value of node."""
        return self.value_sum / self.visits if self.visits > 0 else 0.0
    
    def ucb_score(self, parent_visits: int, c_puct: float = 1.0) -> float:
        """Upper confidence bound score for selection."""
        if self.visits == 0:
            return float('inf')
        
        exploration = c_puct * math.sqrt(2 * math.log(parent_visits) / self.visits)
        return self.value + exploration


# ========================= FLAT TENSOR TREE (HOTEL ROOMS) =========================

class FlatTensorTree:
    """Flat tensor-based tree structure - think of it as a hotel with numbered rooms.
    
    Each node is a room number (index), and we store all node properties in 
    parallel arrays indexed by room number.
    """
    
    def __init__(self, max_nodes: int, board_size: int, device: torch.device):
        self.max_nodes = max_nodes
        self.board_size = board_size
        self.device = device
        self.num_positions = board_size * board_size + 1  # +1 for pass
        
        # Hotel guest registry - parallel arrays for node properties
        self.visits = torch.zeros(max_nodes, dtype=torch.float32, device=device)
        self.value_sum = torch.zeros(max_nodes, dtype=torch.float32, device=device)
        self.is_expanded = torch.zeros(max_nodes, dtype=torch.bool, device=device)
        
        # Hotel room connections - children[room_number, action_slot] = child_room_number
        self.children = torch.full((max_nodes, self.num_positions), -1, dtype=torch.int32, device=device)
        
        # Parent tracking for easier navigation
        self.parents = torch.full((max_nodes,), -1, dtype=torch.int32, device=device)
        self.parent_move = torch.full((max_nodes, 2), -1, dtype=torch.int32, device=device)
        
        # Room allocation counter
        self.next_room = 1  # Room 0 is always root
        
        # Initialize root
        self.is_expanded[0] = True
    
    def allocate_room(self, parent_idx: int, move: Tensor) -> int:
        """Allocate a new room (node) in the hotel.
        
        Returns room number without using .item()
        """
        if self.next_room >= self.max_nodes:
            raise RuntimeError("Hotel is full!")
        
        room_idx = self.next_room
        self.next_room += 1
        
        # Register parent-child relationship
        self.parents[room_idx] = parent_idx
        self.parent_move[room_idx] = move
        
        # Calculate action slot for this move
        if move[0] == -1:  # Pass move
            slot = self.board_size * self.board_size
        else:
            slot = move[0] * self.board_size + move[1]
        
        # Register in parent's children
        self.children[parent_idx, slot] = room_idx
        
        return room_idx
    
    def select_best_child_ucb(self, parent_idx: int, c_puct: float) -> Tuple[Tensor, Tensor]:
        """Select best child using fully vectorized UCB calculation.
        
        Returns (move, child_idx) as tensors without .item() calls.
        """
        # Get all children room numbers
        children_indices = self.children[parent_idx]  # Shape: (num_positions,)
        valid_mask = children_indices >= 0
        
        if not valid_mask.any():
            # No children - return pass move and invalid index
            return create_pass_positions(1, self.device)[0], torch.tensor(-1, device=self.device)
        
        valid_indices = children_indices[valid_mask]
        
        # Vectorized UCB calculation for all children at once
        child_visits = self.visits[valid_indices]
        child_values = torch.where(
            child_visits > 0,
            self.value_sum[valid_indices] / child_visits,
            torch.zeros_like(child_visits)
        )
        
        parent_visits = self.visits[parent_idx]
        exploration = c_puct * torch.sqrt(2 * torch.log(parent_visits + 1) / (child_visits + 1e-8))
        ucb_scores = child_values + exploration
        ucb_scores = torch.where(child_visits == 0, float('inf'), ucb_scores)
        
        # Find best child
        best_idx = ucb_scores.argmax()
        best_child_idx = valid_indices[best_idx]
        
        # Convert action slot back to move
        slots = valid_mask.nonzero().squeeze(-1)
        best_slot = slots[best_idx]
        
        if best_slot == self.num_positions - 1:
            move = create_pass_positions(1, self.device)[0]
        else:
            row = best_slot // self.board_size
            col = best_slot % self.board_size
            move = torch.stack([row, col])
        
        return move, best_child_idx
    
    def get_visit_distribution(self, node_idx: int) -> Tensor:
        """Get visit counts for all children of a node.
        
        Returns tensor of visits without using .item().
        """
        children_indices = self.children[node_idx]
        valid_mask = children_indices >= 0
        
        # Create visit distribution
        visit_dist = torch.zeros(self.num_positions, device=self.device)
        if valid_mask.any():
            valid_indices = children_indices[valid_mask]
            visit_dist[valid_mask] = self.visits[valid_indices]
        
        return visit_dist
    
    def backup_value(self, path_indices: Tensor, value: float):
        """Backup value through a path of nodes.
        
        Updates all nodes in the path simultaneously.
        """
        # Filter valid indices
        valid_mask = path_indices >= 0
        valid_indices = path_indices[valid_mask]
        
        if valid_indices.numel() > 0:
            # Update visits and values for all nodes in path
            self.visits[valid_indices] += 1
            
            # Alternate sign as we go up the tree
            signs = torch.pow(-1.0, torch.arange(len(valid_indices), device=self.device))
            self.value_sum[valid_indices] += value * signs[:valid_indices.numel()]


# ========================= SIMPLE MCTS IMPLEMENTATION =========================

class SimpleMCTS:
    """Simple MCTS implementation using flat tensor tree structure.
    
    Think of the tree as a hotel where each node is a room with a number.
    We never need to use .item() because we work with room numbers as tensors.
    """
    
    def __init__(
        self,
        simulations: int = 100,
        c_puct: float = 1.41,  # sqrt(2) is traditional for UCB1
        device: Optional[torch.device] = None,
        max_rollout_depth: int = 200
    ):
        """Initialize MCTS."""
        self.simulations = simulations
        self.c_puct = c_puct
        self.device = device or select_device()
        self.max_rollout_depth = max_rollout_depth
        
        # Will be initialized per search
        self._tree = None
        self._board_size = None
    
    def select_move(self, board: TensorBoard, temperature: float = 1.0) -> Tensor:
        """Select best move using MCTS.
        
        Args:
            board: Current board state (single game)
            temperature: Sampling temperature (0 = greedy)
            
        Returns:
            Move tensor of shape (2,) with [row, col]
        """
        # Initialize flat tensor tree (our hotel)
        self._board_size = board.board_size
        self._tree = FlatTensorTree(
            max_nodes=self.simulations * 30,  # Approximate max nodes
            board_size=board.board_size,
            device=self.device
        )
        
        # Run simulations - root is always at index 0
        for _ in range(self.simulations):
            board_copy = self._copy_board(board)
            
            # Track path as room numbers
            path = torch.full((50,), -1, dtype=torch.int32, device=self.device)
            path_length = 0
            
            value = self._simulate_flat(board_copy, 0, path, path_length)
            
            # Backup through path
            self._tree.backup_value(path, value)
        
        # Select move based on visit counts
        return self._choose_move_flat(0, temperature)
    
    def _simulate_flat(self, board: TensorBoard, node_idx: int, path: Tensor, path_length: int) -> float:
        """Run one simulation using flat tree structure.
        
        Args:
            board: Current board state (will be modified)
            node_idx: Current node index (room number)
            path: Tensor tracking path indices
            path_length: Current path length
            
        Returns:
            Value estimate for state
        """
        # Add current node to path
        path[path_length] = node_idx
        path_length += 1
        
        # Check if game over
        if board.is_game_over()[0]:
            return self._evaluate_terminal_flat(board).item()
        
        # Expand if needed
        if not self._tree.is_expanded[node_idx]:
            return self._expand_and_evaluate_flat(board, node_idx)
        
        # Select best child
        move, child_idx = self._tree.select_best_child_ucb(node_idx, self.c_puct)
        
        # If no valid child, treat as terminal
        if (child_idx == -1).all():
            return self._evaluate_terminal_flat(board).item()
        
        # Make move
        board.step(move.unsqueeze(0))
        
        # Recurse
        return -self._simulate_flat(board, child_idx.item(), path, path_length)
    
    def _expand_and_evaluate_flat(self, board: TensorBoard, node_idx: int) -> float:
        """Expand node and evaluate state with random rollout.
        
        Uses flat tree structure without creating MCTSNode objects.
        """
        self._tree.is_expanded[node_idx] = True
        
        # Get legal moves
        legal_moves = board.legal_moves()[0]  # Single game
        
        # Pre-allocate children for all legal moves
        if legal_moves.any():
            slots = legal_moves.nonzero()
            for i in range(slots.shape[0]):
                move = slots[i]
                self._tree.allocate_room(node_idx, move)
        else:
            # Only pass move available
            pass_move = create_pass_positions(1, self.device)[0]
            self._tree.allocate_room(node_idx, pass_move)
        
        # Evaluate with random rollout
        board_copy = self._copy_board(board)
        return self._random_rollout_flat(board_copy).item()
    
    def _choose_move_flat(self, root_idx: int, temperature: float) -> Tensor:
        """Choose final move based on visit counts using flat structure."""
        # Get visit distribution for root's children
        visit_dist = self._tree.get_visit_distribution(root_idx)
        
        if temperature == 0:
            # Greedy selection
            best_slot = visit_dist.argmax()
        else:
            # Temperature-based sampling
            if visit_dist.sum() == 0:
                # Uniform random
                best_slot = torch.randint(visit_dist.numel(), (1,), device=self.device)[0]
            else:
                probs = torch.pow(visit_dist, 1.0 / temperature)
                probs = probs / probs.sum()
                best_slot = torch.multinomial(probs, 1)[0]
        
        # Convert action slot to move
        if best_slot == self._board_size * self._board_size:
            return create_pass_positions(1, self.device)[0]
        
        row = best_slot // self._board_size
        col = best_slot % self._board_size
        return torch.stack([row, col])
    
    def _evaluate_terminal_flat(self, board: TensorBoard) -> Tensor:
        """Evaluate terminal state purely with tensor operations."""
        scores = board.compute_scores()[0]
        score_diff = scores[0] - scores[1]  # black - white
        
        # Create evaluation tensor based on current player
        # Returns 1.0 if current player wins, -1.0 if loses, 0.0 for draw
        current_player = board.current_player[0]
        
        # Fully vectorized evaluation
        eval_tensor = torch.sign(score_diff) * (1 - 2 * current_player.float())
        eval_tensor = torch.where(score_diff == 0, torch.tensor(0.0, device=self.device), eval_tensor)
        
        return eval_tensor
    
    def _random_rollout_flat(self, board: TensorBoard) -> Tensor:
        """Perform random rollout using only tensor operations."""
        for _ in range(self.max_rollout_depth):
            if board.is_game_over()[0]:
                break
            
            # Get legal moves
            legal = board.legal_moves()[0]
            
            if legal.any():
                # Random legal move using tensor operations
                flat_legal = legal.flatten().unsqueeze(0)
                flat_idx = sample_from_mask(flat_legal)[0]
                row, col = flat_to_2d(flat_idx.unsqueeze(0), board.board_size)
                move = torch.stack([row[0], col[0]]).unsqueeze(0)
            else:
                # Pass if no legal moves
                move = create_pass_positions(1, self.device)
            
            board.step(move)
        
        return self._evaluate_terminal_flat(board)
    
    def _copy_board(self, board: TensorBoard) -> TensorBoard:
        """Create a copy of the board for simulation."""
        new_board = TensorBoard(1, board.board_size, board.device)
        
        # Copy state tensors
        new_board.stones.copy_(board.stones)
        new_board.current_player.copy_(board.current_player)
        new_board.position_hash.copy_(board.position_hash)
        new_board.ko_points.copy_(board.ko_points)
        new_board.pass_count.copy_(board.pass_count)
        
        return new_board
    
    # ==================== COMPATIBILITY METHODS ====================
    # These methods maintain backward compatibility with old interface
    
    def _simulate(self, board: TensorBoard, node: MCTSNode) -> float:
        """Compatibility wrapper for old interface."""
        # Just use the flat version starting from root
        path = torch.full((50,), -1, dtype=torch.int32, device=self.device)
        return self._simulate_flat(board, 0, path, 0)
    
    def _choose_move(self, root: MCTSNode, temperature: float) -> Tensor:
        """Compatibility wrapper for old interface."""
        # Just use the flat version with root index 0
        return self._choose_move_flat(0, temperature)