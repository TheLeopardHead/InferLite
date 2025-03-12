import torch
import logging
from typing import Set
from .base_sampler import Sampler

# Create module-level logger
logger = logging.getLogger(__name__)

class RepetitionPenaltySampler(Sampler):
    """Wrapper sampler that adds repetition penalty functionality"""
    
    def __init__(self, base_sampler: Sampler, penalty: float = 1.0):
        """
        Initialize repetition penalty sampler
        Args:
            base_sampler: Base sampler to wrap
            penalty: Penalty coefficient, 1.0 means no penalty
        """
        super().__init__()
        self.base_sampler = base_sampler
        self.penalty = penalty
        logger.debug(f"Initializing RepetitionPenalty sampler: penalty={penalty}")
    
    def sample(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply repetition penalty and sample next token
        Args:
            logits: Model output logits, shape [batch_size, vocab_size]
            input_ids: Current input sequence, used to identify generated tokens
        Returns:
            next_token: Sampled next token, shape [batch_size, 1]
        """
        # Apply repetition penalty
        if self.penalty != 1.0:
            # Get set of already generated tokens
            seen_tokens = set(input_ids[0].tolist())
            
            # Apply penalty to each already generated token
            for token_id in seen_tokens:
                if logits[0, token_id] < 0:
                    # For negative values, multiply by penalty (make more negative)
                    logits[0, token_id] *= self.penalty
                else:
                    # For positive values, divide by penalty (make smaller)
                    logits[0, token_id] /= self.penalty
            
            logger.debug(f"Applied repetition penalty {self.penalty} to {len(seen_tokens)} tokens")
        
        # Use base sampler to sample
        return self.base_sampler.sample(logits)
    
    def update_state(self, input_ids: torch.Tensor) -> None:
        """
        Update sampler internal state
        Args:
            input_ids: Current input sequence, including newly generated token
        """
        # Update base sampler state (if any)
        self.base_sampler.update_state(input_ids) 