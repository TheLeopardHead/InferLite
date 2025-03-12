import torch
import torch.nn as nn
import math
import logging
from typing import Tuple

# Create module-level logger
logger = logging.getLogger(__name__)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        """
        Initialize rotary position encoding
        Args:
            dim: Encoding dimension
            max_position_embeddings: Maximum position encoding length
            base: Base value
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Calculate frequencies for position encoding
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Generate position encoding
        self._build_embeddings()
        logger.debug(f"Initializing RotaryEmbedding: dim={dim}, max_position_embeddings={max_position_embeddings}, base={base}")
        
    def _build_embeddings(self):
        """Generate position encoding matrix"""
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])  # [1, 1, max_position_embeddings, dim]
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
        logger.debug(f"Building position encoding cache: shape={self.cos_cached.shape}")
        
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get position encoding for specified length
        Args:
            x: Input tensor
            seq_len: Sequence length
        Returns:
            cos: Cosine position encoding
            sin: Sine position encoding
        """
        # Check if sequence length exceeds maximum position encoding length
        if seq_len > self.max_position_embeddings:
            logger.warning(f"Requested sequence length ({seq_len}) exceeds maximum position encoding length ({self.max_position_embeddings}), will be truncated")
            seq_len = self.max_position_embeddings
        
        # Ensure returned encoding length matches requested sequence length
        try:
            # Get cached position encoding
            if not hasattr(self, "cos_cached") or not hasattr(self, "sin_cached"):
                logger.warning("Position encoding cache does not exist, rebuilding position encoding")
                self._build_embeddings()
            
            # Ensure cached position encoding shape is correct
            if self.cos_cached.size(2) < seq_len:
                logger.warning(f"Cached position encoding length ({self.cos_cached.size(2)}) is less than requested sequence length ({seq_len}), rebuilding position encoding")
                # Save original max_position_embeddings
                orig_max_pos = self.max_position_embeddings
                # Update max_position_embeddings
                self.max_position_embeddings = max(seq_len, orig_max_pos * 2)
                logger.info(f"Updating maximum position encoding length: {orig_max_pos} -> {self.max_position_embeddings}")
                # Rebuild position encoding
                self._build_embeddings()
            
            # Get position encoding for specified length
            cos = self.cos_cached[:, :, :seq_len, :].to(dtype=x.dtype)
            sin = self.sin_cached[:, :, :seq_len, :].to(dtype=x.dtype)
            
            # Ensure shape is correct
            if cos.size(2) != seq_len:
                logger.error(f"Position encoding length mismatch: cos.size(2)={cos.size(2)}, seq_len={seq_len}")
                # If position encoding length is greater than sequence length, truncate
                if cos.size(2) > seq_len:
                    cos = cos[:, :, :seq_len, :]
                    sin = sin[:, :, :seq_len, :]
                # If position encoding length is less than sequence length, pad
                else:
                    # Create new position encoding tensors
                    new_cos = torch.zeros(cos.size(0), cos.size(1), seq_len, cos.size(3), device=cos.device, dtype=cos.dtype)
                    new_sin = torch.zeros(sin.size(0), sin.size(1), seq_len, sin.size(3), device=sin.device, dtype=sin.dtype)
                    # Copy position encoding
                    new_cos[:, :, :cos.size(2), :] = cos
                    new_sin[:, :, :sin.size(2), :] = sin
                    # For positions beyond the original length, use the last position encoding
                    if cos.size(2) > 0:
                        new_cos[:, :, cos.size(2):, :] = cos[:, :, -1:, :]
                        new_sin[:, :, sin.size(2):, :] = sin[:, :, -1:, :]
                    cos = new_cos
                    sin = new_sin
            
            logger.debug(f"Returning position encoding: cos.shape={cos.shape}, sin.shape={sin.shape}, seq_len={seq_len}")
            return (cos, sin)
        except Exception as e:
            logger.error(f"Error getting position encoding: {e}")
            # If error, try to create a zero tensor matching the requested sequence length
            device = x.device if hasattr(x, 'device') else 'cpu'
            dtype = x.dtype if hasattr(x, 'dtype') else torch.float32
            
            # Create all-zero position encoding
            cos = torch.ones((1, 1, seq_len, self.dim), device=device, dtype=dtype)
            sin = torch.zeros((1, 1, seq_len, self.dim), device=device, dtype=dtype)
            
            logger.warning(f"Returning emergency position encoding: cos.shape={cos.shape}, sin.shape={sin.shape}")
            return (cos, sin) 