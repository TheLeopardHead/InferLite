import torch
import torch.nn.functional as F
import logging
from .base_sampler import Sampler

# Create module-level logger
logger = logging.getLogger(__name__)

class TopKTopPSampler(Sampler):
    """Sampler combining top-k and top-p (nucleus) sampling"""
    
    def __init__(self, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0):
        """
        Initialize TopK-TopP sampler
        Args:
            temperature: Temperature parameter, controls distribution smoothness
            top_k: Only keep top k tokens with highest probability, 0 means disabled
            top_p: Only keep tokens with cumulative probability up to p, 1.0 means disabled
        """
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        logger.debug(f"Initializing TopKTopP sampler: temperature={temperature}, top_k={top_k}, top_p={top_p}")
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample next token from logits
        Args:
            logits: Model output logits, shape [batch_size, vocab_size]
        Returns:
            next_token: Sampled next token, shape [batch_size, 1]
        """
        # Apply temperature
        if self.temperature > 0:
            logits = logits / self.temperature
            logger.debug(f"Applied temperature {self.temperature}")
        
        # Apply top-k filtering
        if self.top_k > 0:
            # Get top-k values and indices
            top_k_values, _ = torch.topk(logits, k=self.top_k)
            # Get threshold (k-th largest value in each row)
            filter_value = top_k_values[:, -1].unsqueeze(-1)
            # Set logits below threshold to negative infinity
            logits = torch.where(logits < filter_value, 
                                torch.full_like(logits, float('-inf')), 
                                logits)
            logger.debug(f"Applied top-k filtering with k={self.top_k}")
        
        # Apply top-p sampling
        if self.top_p < 1.0:
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            # Calculate cumulative probabilities
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above top_p
            sorted_indices_to_remove = cumulative_probs > self.top_p
            # Shift indices to remove all tokens after the first one exceeding threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            # Keep at least one token (don't remove the first token)
            sorted_indices_to_remove[..., 0] = 0
            
            # Map indices_to_remove back to original order
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
            logger.debug(f"Applied top-p filtering with p={self.top_p}")
        
        # Calculate probability distribution
        probs = F.softmax(logits, dim=-1)
        
        # Sample
        next_token = torch.multinomial(probs, num_samples=1)
        logger.debug(f"Sampled token: {next_token.item()}")
        
        return next_token 