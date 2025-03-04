from typing import List, Optional, Union
import os
import logging
from transformers import AutoTokenizer

# Create module-level logger
logger = logging.getLogger(__name__)

class Tokenizer:
    def __init__(self, model_path: str):
        """
        Initialize tokenizer
        Args:
            model_path: Model or tokenizer path (can be a Hugging Face model ID or local path)
        """
        if os.path.exists(model_path):
            # If it's a directory path, try to load the tokenizer from the directory
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"Loaded tokenizer from local path: {model_path}")
        else:
            # Otherwise try to load from Hugging Face
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                logger.info(f"Loaded tokenizer from Hugging Face: {model_path}")
            except Exception as e:
                logger.error(f"Unable to load tokenizer: {e}")
                raise ValueError(f"Unable to load tokenizer: {e}")
        
        # Get special token IDs
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        
        logger.info(f"Tokenizer loaded successfully, vocabulary size: {len(self.tokenizer)}")
        logger.info(f"Special tokens: BOS={self.bos_token_id}, EOS={self.eos_token_id}, PAD={self.pad_token_id}")
    
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """
        Encode text to token ids
        Args:
            text: Input text
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token
        Returns:
            token_ids: List of token ids
        """
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=True,  # This adds special tokens according to the model configuration
            return_tensors=None,  # Return Python list instead of tensor
        )
        logger.debug(f"Encoded text, input length: {len(text)}, output token count: {len(token_ids)}")
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token ids to text
        Args:
            token_ids: List of token ids
        Returns:
            text: Decoded text
        """
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        logger.debug(f"Decoded tokens, input token count: {len(token_ids)}, output text length: {len(text)}")
        return text
    
    def __len__(self) -> int:
        """Return vocabulary size"""
        return len(self.tokenizer) 