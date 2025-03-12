from abc import ABC, abstractmethod
import torch
import logging

# Create module-level logger
logger = logging.getLogger(__name__)

class Sampler(ABC):
    """Base sampler class, defines sampling interface"""
    
    def __init__(self):
        """Initialize sampler"""
        logger.debug("Initializing base sampler")
    
    @abstractmethod
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample next token from logits
        Args:
            logits: Model output logits, shape [batch_size, vocab_size]
        Returns:
            next_token: Sampled next token, shape [batch_size, 1]
        """
        pass
    
    def update_state(self, input_ids: torch.Tensor) -> None:
        """
        Update sampler internal state (if any)
        Args:
            input_ids: Current input sequence, including newly generated token
        """
        # Default implementation is empty, subclasses can override as needed
        pass 