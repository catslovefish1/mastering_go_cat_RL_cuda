# Optimized version of the super-ko filtering with better memory efficiency
import torch
from torch import Tensor
from typing import Dict, Tuple

class OptimizedSuperKo:
    """Optimized methods for super-ko detection with reduced memory usage."""
    
    @staticmethod
    def compute_capture_delta_sparse(
        cap_mask: Tensor,  # (B, N2, N2) bool
        zobrist_transposed: Tensor,  # (3, N2) int32
        opponent_indices: Tensor,  # (B,) int64
        device: torch.device
    ) -> Tensor:
        """
        Compute capture delta using sparse operations to avoid B*N2*N2 dense tensors.
        
        Returns:
            cap_delta: (B, N2) int32 - XOR reduction of captured stones for each candidate
        """
        B, N2, _ = cap_mask.shape
        
        # Get zobrist values for opponent->empty transitions
        Z_opp = zobrist_transposed[opponent_indices + 1]  # (B, N2) int32
        Z_emp = zobrist_transposed[0].expand(B, -1)  # (B, N2) int32
        D = torch.bitwise_xor(Z_opp, Z_emp)  # (B, N2) int32 - per-cell toggle values
        
        # Find non-zero capture positions (sparse representation)
        batch_idx, cand_idx, pos_idx = cap_mask.nonzero(as_tuple=True)
        
        if batch_idx.numel() == 0:
            # No captures at all
            return torch.zeros(B, N2, dtype=torch.int32, device=device)
        
        # Initialize result
        cap_delta = torch.zeros(B, N2, dtype=torch.int32, device=device)
        
        # Get the zobrist toggle values for captured positions
        values = D[batch_idx, pos_idx]  # (nnz,) int32
        
        # Create flat indices for accumulation
        flat_idx = batch_idx * N2 + cand_idx  # (nnz,)
        
        # Use scatter with custom reduction for XOR
        # Note: PyTorch doesn't have native scatter_xor, so we'll group and reduce
        unique_idx = torch.unique(flat_idx)
        
        for idx in unique_idx:
            mask = (flat_idx == idx)
            if mask.any():
                # XOR all values for this (batch, candidate) pair
                xor_result = values[mask][0]
                for v in values[mask][1:]:
                    xor_result = torch.bitwise_xor(xor_result, v)
                
                # Unpack flat index
                b = idx // N2
                c = idx % N2
                cap_delta[b, c] = xor_result
        
        return cap_delta
    
    @staticmethod  
    def compute_capture_delta_chunked(
        cap_mask: Tensor,  # (B, N2, N2) bool
        zobrist_transposed: Tensor,  # (3, N2) int32
        opponent_indices: Tensor,  # (B,) int64
        chunk_size: int = 64
    ) -> Tensor:
        """
        Compute capture delta by processing candidates in chunks.
        
        Returns:
            cap_delta: (B, N2) int32
        """
        B, N2, _ = cap_mask.shape
        device = cap_mask.device
        
        # Zobrist deltas for opponent->empty
        Z_opp = zobrist_transposed[opponent_indices + 1]  # (B, N2) int32
        Z_emp = zobrist_transposed[0].expand(B, -1)  # (B, N2) int32
        D = torch.bitwise_xor(Z_opp, Z_emp)  # (B, N2) int32
        
        # Initialize result
        cap_delta = torch.zeros(B, N2, dtype=torch.int32, device=device)
        
        # Process candidates in chunks to reduce memory
        for start in range(0, N2, chunk_size):
            end = min(start + chunk_size, N2)
            
            # Get captures for this chunk of candidates
            chunk_mask = cap_mask[:, start:end, :]  # (B, chunk_size, N2)
            
            # For each candidate in chunk, XOR-reduce captured positions
            for i in range(end - start):
                captures = chunk_mask[:, i, :]  # (B, N2)
                
                # Only process batches that have captures for this candidate
                has_captures = captures.any(dim=1)  # (B,)
                if has_captures.any():
                    # Masked XOR reduction
                    masked_D = torch.where(captures, D, torch.zeros_like(D))
                    
                    # XOR reduce along position dimension for active batches
                    for b in has_captures.nonzero(as_tuple=True)[0]:
                        active_positions = captures[b].nonzero(as_tuple=True)[0]
                        if active_positions.numel() > 0:
                            xor_val = D[b, active_positions[0]]
                            for pos in active_positions[1:]:
                                xor_val = torch.bitwise_xor(xor_val, D[b, pos])
                            cap_delta[b, start + i] = xor_val
        
        return cap_delta
    
    @staticmethod
    def xor_reduce_optimized(x: Tensor, dim: int = -1) -> Tensor:
        """
        Optimized XOR reduction using reshape instead of slicing.
        
        Args:
            x: Input tensor
            dim: Dimension to reduce (must be last dim for now)
            
        Returns:
            Reduced tensor with dim squeezed
        """
        if dim != -1 and dim != x.ndim - 1:
            raise NotImplementedError("Only last dimension reduction supported")
            
        *batch_dims, K = x.shape
        
        # Early exit for single element
        if K == 1:
            return x.squeeze(-1)
        
        # Pad to power of 2
        P = 1 << (K - 1).bit_length()
        if P != K:
            padding_shape = (*batch_dims, P - K)
            padding = torch.zeros(padding_shape, dtype=x.dtype, device=x.device)
            x = torch.cat([x, padding], dim=-1)
        
        # Reshape and reduce
        x = x.reshape(*batch_dims, P)
        while x.shape[-1] > 1:
            x = x.reshape(*batch_dims, -1, 2)
            x = torch.bitwise_xor(x[..., 0], x[..., 1])
        
        return x.squeeze(-1)
    
    @staticmethod
    def filter_super_ko_optimized(
        legal_mask: Tensor,  # (B, H, W) bool
        capture_stone_mask: Tensor,  # (B, N2, N2) bool  
        current_hash: Tensor,  # (B,) int32
        hash_history: Tensor,  # (B, max_moves) int32
        move_count: Tensor,  # (B,) int16
        zobrist_table: Tensor,  # (N2, 3) int32
        zobrist_transposed: Tensor,  # (3, N2) int32
        current_player: Tensor,  # (B,) int8
        use_sparse: bool = True,
        history_chunk_size: int = 64
    ) -> Tensor:
        """
        Optimized super-ko filtering with better memory efficiency.
        
        Returns:
            filtered_mask: (B, H, W) bool - legal moves excluding super-ko violations
        """
        B, H, W = legal_mask.shape
        N2 = H * W
        device = legal_mask.device
        
        # Flatten legal mask
        legal_flat = legal_mask.view(B, N2).bool()
        
        # Player indices (ensure int32/int64 as needed)
        player = current_player.long()  # (B,)
        opponent = 1 - player  # (B,)
        
        # === Placement Delta (already efficient) ===
        lin_idx = torch.arange(N2, device=device, dtype=torch.int64)
        place_old = zobrist_table[lin_idx, 0]  # (N2,) int32 - empty state
        
        # Vectorized placement new values
        place_new_black = zobrist_table[lin_idx, 1]  # (N2,) int32
        place_new_white = zobrist_table[lin_idx, 2]  # (N2,) int32
        
        # Select based on current player
        place_new = torch.where(
            player[:, None] == 0,  # (B, 1)
            place_new_black[None, :],  # broadcast to (B, N2)
            place_new_white[None, :]
        )  # (B, N2) int32
        
        # XOR for placement delta
        place_delta = torch.bitwise_xor(
            place_old[None, :].expand(B, -1),
            place_new
        )  # (B, N2) int32
        
        # === Capture Delta (optimized) ===
        if use_sparse:
            cap_delta = OptimizedSuperKo.compute_capture_delta_sparse(
                capture_stone_mask, zobrist_transposed, opponent, device
            )
        else:
            cap_delta = OptimizedSuperKo.compute_capture_delta_chunked(
                capture_stone_mask, zobrist_transposed, opponent
            )
        
        # === Compute New Hashes ===
        new_hash = current_hash[:, None] ^ place_delta ^ cap_delta  # (B, N2) int32
        
        # === History Comparison (chunked) ===
        M = hash_history.shape[1]
        hist_mask = torch.arange(M, device=device) < move_count[:, None]  # (B, M)
        
        is_repeat = torch.zeros(B, N2, dtype=torch.bool, device=device)
        
        # Process history in chunks
        for start in range(0, M, history_chunk_size):
            end = min(start + history_chunk_size, M)
            
            # Get chunk of history
            hist_chunk = hash_history[:, start:end]  # (B, chunk)
            mask_chunk = hist_mask[:, start:end]  # (B, chunk)
            
            # Compare: (B, N2, 1) == (B, 1, chunk) -> (B, N2, chunk)
            matches = (new_hash[:, :, None] == hist_chunk[:, None, :])
            matches = matches & mask_chunk[:, None, :]  # Apply validity mask
            
            # Accumulate any matches
            is_repeat |= matches.any(dim=2)
        
        # Only check candidates that are legal
        is_repeat &= legal_flat
        
        # Final mask: legal but not repeating
        repeat_mask = is_repeat.view(B, H, W)
        return legal_mask & ~repeat_mask


# Example of how to integrate into TensorBoard class
def create_optimized_filter_method(self):
    """
    Drop-in replacement for _filter_super_ko_vectorized method.
    """
    def _filter_super_ko_vectorized(legal_mask: Tensor, info: Dict) -> Tensor:
        return OptimizedSuperKo.filter_super_ko_optimized(
            legal_mask=legal_mask,
            capture_stone_mask=info["capture_stone_mask"],
            current_hash=self.current_hash,
            hash_history=self.hash_history,
            move_count=self.move_count,
            zobrist_table=self.Zpos,  # (N2, 3)
            zobrist_transposed=self.ZposT,  # (3, N2)
            current_player=self.current_player,
            use_sparse=True,  # Use sparse method for large boards
            history_chunk_size=64
        )
    
    return _filter_super_ko_vectorized
