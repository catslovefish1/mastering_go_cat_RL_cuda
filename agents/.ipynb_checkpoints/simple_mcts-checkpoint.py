"""agents/simple_mcts.py - Simple MCTS-based Go agent.

This agent uses Monte Carlo Tree Search with random rollouts for move selection.
No neural networks, just pure tree search and random playouts.
"""

from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor

# Import engine
from engine.tensor_native import TensorBoard

# Import simple MCTS implementation
from NN.simple_mcts import SimpleMCTS

# Import shared utilities
from utils.shared import (
    select_device,
    create_pass_positions
)


class SimpleMCTSAgent:
    """Simple MCTS agent using only random rollouts.
    
    This agent uses Monte Carlo Tree Search to select moves,
    evaluating positions through random playouts to the end of the game.
    
    Attributes:
        device: Computation device (CUDA/MPS/CPU)
        mcts: SimpleMCTS instance for tree search
        simulations: Number of simulations per move
    """
    
    def __init__(
        self,
        simulations: int = 100,
        c_puct: float = 1.41,  # sqrt(2) for UCB1
        max_rollout_depth: int = 200,
        device: Optional[torch.device | str] = None
    ):
        """Initialize simple MCTS agent.
        
        Args:
            simulations: Number of MCTS simulations per move
            c_puct: Exploration constant for UCB formula (sqrt(2) is traditional)
            max_rollout_depth: Maximum moves in random rollouts
            device: Target device for computations
        """
        self.device = (
            torch.device(device) if device is not None 
            else select_device()
        )
        
        # Create MCTS instance
        self.mcts = SimpleMCTS(
            simulations=simulations,
            c_puct=c_puct,
            device=self.device,
            max_rollout_depth=max_rollout_depth
        )
        
        self.simulations = simulations
    
    def select_moves(self, boards: TensorBoard, temperature: float = 1.0) -> Tensor:
        """Select moves for all games in batch using MCTS.
        
        Note: This implementation processes games sequentially.
        
        Args:
            boards: TensorBoard instance with current game states
            temperature: Sampling temperature (0 = greedy, 1 = proportional to visits)
            
        Returns:
            Tensor of shape (B, 2) with [row, col] coordinates.
            Returns [-1, -1] for pass moves.
        """
        batch_size = boards.batch_size
        
        # Initialize moves as passes
        moves = create_pass_positions(batch_size, self.device).to(torch.int32)
        
        # Process each game
        for i in range(batch_size):
            if not boards.is_game_over()[i]:
                # Extract single game state
                single_board = self._extract_single_board(boards, i)
                
                # Run MCTS to select move
                move = self.mcts.select_move(single_board, temperature)
                moves[i] = move.to(torch.int32)
        
        return moves
    
    def _extract_single_board(self, boards: TensorBoard, index: int) -> TensorBoard:
        """Extract a single board from batch for MCTS processing.
        
        Args:
            boards: Batch of boards
            index: Index of board to extract
            
        Returns:
            Single TensorBoard instance
        """
        single_board = TensorBoard(1, boards.board_size, self.device)
        
        # Copy state for specific game
        single_board.stones[0] = boards.stones[index]
        single_board.current_player[0] = boards.current_player[index]
        single_board.position_hash[0] = boards.position_hash[index]
        single_board.ko_points[0] = boards.ko_points[index]
        single_board.pass_count[0] = boards.pass_count[index]
        
        return single_board


# ========================= CONVENIENCE CONSTRUCTORS =========================

def create_simple_mcts_agent(
    simulations: int = 50,
    c_puct: float = 1.41,
    device: Optional[torch.device | str] = None
) -> SimpleMCTSAgent:
    """Create simple MCTS agent using only random rollouts.
    
    Args:
        simulations: Number of simulations per move
        c_puct: Exploration constant (default sqrt(2))
        device: Computation device
        
    Returns:
        SimpleMCTSAgent configured for pure MCTS
    """
    return SimpleMCTSAgent(
        simulations=simulations,
        c_puct=c_puct,  # FIXED: Pass the parameter value, not a type annotation
        device=device
    )


