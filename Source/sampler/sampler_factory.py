import logging
from typing import Dict, Any, Optional
from .base_sampler import Sampler
from .greedy_sampler import GreedySampler
from .top_k_top_p_sampler import TopKTopPSampler
from .repetition_penalty_sampler import RepetitionPenaltySampler

# Create module-level logger
logger = logging.getLogger(__name__)

class SamplerFactory:
    """Factory class for creating samplers"""
    
    @staticmethod
    def create_sampler(config: Dict[str, Any]) -> Sampler:
        """
        Create sampler based on configuration
        Args:
            config: Sampler configuration, containing the following optional fields:
                - do_sample: Whether to use sampling (otherwise use greedy decoding)
                - temperature: Temperature parameter
                - top_k: Top-K parameter
                - top_p: Top-P parameter
                - repetition_penalty: Repetition penalty parameter
        Returns:
            Sampler instance
        """
        # Extract configuration parameters, use default values
        do_sample = config.get('do_sample', True)
        temperature = config.get('temperature', 0.7)
        top_k = config.get('top_k', 0)
        top_p = config.get('top_p', 0.9)
        repetition_penalty = config.get('repetition_penalty', 1.0)
        
        # Create base sampler based on configuration
        if do_sample:
            logger.info(f"Creating sampling-based sampler: temperature={temperature}, top_k={top_k}, top_p={top_p}")
            base_sampler = TopKTopPSampler(temperature=temperature, top_k=top_k, top_p=top_p)
        else:
            logger.info("Creating greedy sampler")
            base_sampler = GreedySampler()
        
        # Add wrapper if repetition penalty is needed
        if repetition_penalty != 1.0:
            logger.info(f"Adding repetition penalty: {repetition_penalty}")
            return RepetitionPenaltySampler(base_sampler, penalty=repetition_penalty)
        
        return base_sampler
    
    @staticmethod
    def create_greedy_sampler() -> GreedySampler:
        """
        Create a greedy sampler that always selects the token with highest probability
        Returns:
            GreedySampler instance
        """
        logger.info("Creating greedy sampler using convenience method")
        return GreedySampler()
    
    @staticmethod
    def create_temperature_sampler(temperature: float = 0.7) -> TopKTopPSampler:
        """
        Create a sampler that uses temperature sampling
        Args:
            temperature: Temperature parameter, controls randomness (lower = less random)
        Returns:
            TopKTopPSampler instance with only temperature applied
        """
        logger.info(f"Creating temperature sampler: temperature={temperature}")
        return TopKTopPSampler(temperature=temperature, top_p=1.0, top_k=0)
    
    @staticmethod
    def create_top_k_sampler(k: int, temperature: float = 1.0) -> TopKTopPSampler:
        """
        Create a sampler that uses top-k sampling
        Args:
            k: Number of highest probability tokens to keep
            temperature: Temperature parameter, controls randomness
        Returns:
            TopKTopPSampler instance with top-k filtering
        """
        logger.info(f"Creating top-k sampler: k={k}, temperature={temperature}")
        return TopKTopPSampler(temperature=temperature, top_k=k, top_p=1.0)
    
    @staticmethod
    def create_nucleus_sampler(p: float, temperature: float = 1.0) -> TopKTopPSampler:
        """
        Create a sampler that uses nucleus (top-p) sampling
        Args:
            p: Cumulative probability threshold
            temperature: Temperature parameter, controls randomness
        Returns:
            TopKTopPSampler instance with nucleus sampling
        """
        logger.info(f"Creating nucleus sampler: p={p}, temperature={temperature}")
        return TopKTopPSampler(temperature=temperature, top_p=p, top_k=0)
    
    @staticmethod
    def create_combined_sampler(temperature: float = 0.7, top_k: int = 0, top_p: float = 0.9) -> TopKTopPSampler:
        """
        Create a sampler that combines temperature, top-k and top-p sampling
        Args:
            temperature: Temperature parameter, controls randomness
            top_k: Number of highest probability tokens to keep (0 to disable)
            top_p: Cumulative probability threshold (1.0 to disable)
        Returns:
            TopKTopPSampler instance with combined sampling strategies
        """
        logger.info(f"Creating combined sampler: temperature={temperature}, top_k={top_k}, top_p={top_p}")
        return TopKTopPSampler(temperature=temperature, top_k=top_k, top_p=top_p)
    
    @staticmethod
    def add_repetition_penalty(base_sampler: Sampler, penalty: float) -> RepetitionPenaltySampler:
        """
        Wrap a sampler with repetition penalty
        Args:
            base_sampler: Base sampler to wrap
            penalty: Penalty coefficient (>1.0 to penalize repetition)
        Returns:
            RepetitionPenaltySampler instance wrapping the base sampler
        """
        logger.info(f"Adding repetition penalty wrapper: penalty={penalty}")
        return RepetitionPenaltySampler(base_sampler, penalty=penalty) 