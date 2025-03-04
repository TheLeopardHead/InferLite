from abc import ABC, abstractmethod
import torch
import logging
from typing import Optional, Dict, Any

# Create module-level logger
logger = logging.getLogger(__name__)

class BaseEngine(ABC):
    """Base inference engine interface"""
    
    def __init__(self):
        logger.debug("Initializing BaseEngine")
    
    @abstractmethod
    def load_model(self, model_path: str, config_path: str) -> None:
        """
        Load model
        Args:
            model_path: Model weights path
            config_path: Model configuration file path
        """
        logger.debug(f"Calling load_model: model_path={model_path}, config_path={config_path}")
        pass
    
    @abstractmethod
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Execute forward inference
        Args:
            input_ids: Input token ids
        Returns:
            logits: Output logits
        """
        logger.debug(f"Calling forward: input_shape={input_ids.shape}")
        pass
    
    @abstractmethod
    def generate(self, prompt: str, max_length: int = 100, **kwargs) -> str:
        """
        Generate text
        Args:
            prompt: Input prompt text
            max_length: Maximum generation length
            **kwargs: Other generation parameters
        Returns:
            response: Generated text
        """
        logger.debug(f"Calling generate: prompt_length={len(prompt)}, max_length={max_length}, kwargs={kwargs}")
        pass 