import torch
import logging
from .base_sampler import Sampler

# Create module-level logger
logger = logging.getLogger(__name__)

class GreedySampler(Sampler):
    """Greedy sampler, always selects the token with highest probability"""
    
    def __init__(self):
        """Initialize greedy sampler"""
        super().__init__()
        logger.debug("Initializing greedy sampler")
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample next token from logits (select the highest probability token)
        Args:
            logits: Model output logits, shape [batch_size, vocab_size]
        Returns:
            next_token: Sampled next token, shape [batch_size, 1]
        """
        logger.debug(f"Greedy sampling from logits shape: {logits.shape}")
        return torch.argmax(logits, dim=-1, keepdim=True) 