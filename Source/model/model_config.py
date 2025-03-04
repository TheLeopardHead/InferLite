from dataclasses import dataclass,field, fields
from typing import Any, Dict, List, Optional, Tuple, Union
import os, json
import logging

# Create module-level logger
logger = logging.getLogger(__name__)

@dataclass
class LlamaConfig:
    # Core architecture parameters
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    
    # Position encoding parameters
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # Model structure parameters
    hidden_act: str = "silu"
    attention_bias: bool = False
    mlp_bias: bool = False
    
    # Model type and special tokens
    model_type: str = "llama"
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: Optional[int] = None  # Allow None for later processing
    
    # Other parameters
    tie_word_embeddings: bool = False
    
    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        
        # Process pad_token_id
        if self.pad_token_id is None:
            # If pad_token_id is not specified, try to use eos_token_id
            if self.eos_token_id is not None:
                self.pad_token_id = self.eos_token_id
                logger.warning(f"pad_token_id not found, using eos_token_id({self.eos_token_id}) instead")
            else:
                # If eos_token_id is also not specified, use default value 0
                self.pad_token_id = 0
                logger.warning(f"Neither pad_token_id nor eos_token_id found, using default value 0")
            
    @classmethod
    def from_json(cls, json_file: str) -> "LlamaConfig":
        """Load configuration from JSON file"""
        logger.info(f"Loading configuration from JSON file: {json_file}")
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_hf_config(cls, hf_config_file: str) -> "LlamaConfig":
        """Create configuration from Hugging Face config file
        Args:
            hf_config_file: Path to Hugging Face config file
        Returns:
            config: LlamaConfig instance
        """
        logger.info(f"Loading configuration from Hugging Face config file: {hf_config_file}")
        with open(hf_config_file, 'r') as f:
            hf_config = json.load(f)
        
        # Map Hugging Face config to our config
        config_dict = {
            # Core architecture parameters
            "vocab_size": hf_config["vocab_size"],
            "hidden_size": hf_config["hidden_size"],
            "intermediate_size": hf_config["intermediate_size"],
            "num_layers": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "num_kv_heads": hf_config.get("num_key_value_heads"),
            "head_dim": None,  # Will be calculated by __post_init__
            "max_position_embeddings": hf_config["max_position_embeddings"],
            "rms_norm_eps": hf_config["rms_norm_eps"],
            
            # Position encoding parameters
            "rope_theta": hf_config.get("rope_theta", 10000.0),
            "rope_scaling": hf_config.get("rope_scaling"),
            
            # Model structure parameters
            "hidden_act": hf_config.get("hidden_act", "silu"),
            "attention_bias": hf_config.get("attention_bias", False),
            "mlp_bias": hf_config.get("mlp_bias", False),
            
            # Model type and special tokens
            "model_type": hf_config.get("model_type", "llama"),
            "bos_token_id": hf_config.get("bos_token_id", 1),
            "eos_token_id": hf_config.get("eos_token_id", 2),
            "pad_token_id": hf_config.get("pad_token_id"),  # Allow None
            
            # Other parameters
            "tie_word_embeddings": hf_config.get("tie_word_embeddings", False),
        }
        
        logger.debug(f"Converted configuration: {config_dict}")
        return cls(**config_dict)
    
    def to_json(self, json_file: str) -> None:
        """Save configuration to JSON file"""
        logger.info(f"Saving configuration to JSON file: {json_file}")
        config_dict = {field.name: getattr(self, field.name) for field in fields(self)}
        with open(json_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {field.name: getattr(self, field.name) for field in fields(self)}
