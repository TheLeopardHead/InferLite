import torch
import torch.nn.functional as F
import os
import logging
from typing import Optional, Dict, Any, List, Union
from .base_engine import BaseEngine
from ..model.llama import LlamaModel
from ..model.model_config import LlamaConfig
from ..tokenizer import Tokenizer
from ..sampler import Sampler, SamplerFactory, RepetitionPenaltySampler

# Create module-level logger
logger = logging.getLogger(__name__)

class LLMEngine(BaseEngine):
    def __init__(self, model_path: str, config_path: str, tokenizer_path: str = None, device: str = "cuda"):
        """
        Initialize LLM inference engine
        Args:
            model_path: Model weights path
            config_path: Model configuration file path
            tokenizer_path: Tokenizer model path, if None, use the parent directory of model_path
            device: Running device
        """
        self.device = device
        
        # If tokenizer path is not provided, use the parent directory of the model path
        if tokenizer_path is None:
            tokenizer_path = os.path.dirname(model_path)
            logger.info(f"Tokenizer path not provided, using model directory: {tokenizer_path}")
        
        # Load configuration first to get special token IDs
        logger.info(f"Loading configuration file: {config_path}")
        self.config = LlamaConfig.from_json(config_path)
        
        # Load tokenizer
        self.tokenizer = Tokenizer(tokenizer_path)
        logger.info(f"Special token IDs - Config: BOS={self.config.bos_token_id}, EOS={self.config.eos_token_id}, PAD={self.config.pad_token_id}")
        logger.info(f"Special token IDs - Tokenizer: BOS={self.tokenizer.bos_token_id}, EOS={self.tokenizer.eos_token_id}, PAD={self.tokenizer.pad_token_id}")
        
        # Load model
        self.load_model(model_path)
        
    def load_model(self, model_path: str) -> None:
        """Load model"""
        self.model = LlamaModel(self.config).to(self.device)
        
        logger.info(f"Loading model weights: {model_path}")
        # Check file extension
        if model_path.endswith('.safetensors'):
            try:
                from safetensors import safe_open
                from safetensors.torch import load_file
                
                # Load weights in safetensors format
                state_dict = load_file(model_path)
                logger.info(f"Loading model using safetensors")
            except ImportError:
                logger.error("Need to install safetensors library to load .safetensors format models")
                raise ImportError("Need to install safetensors library to load .safetensors format models")
        else:
            # Load weights in PyTorch format
            state_dict = torch.load(model_path, map_location=self.device)
            logger.info(f"Loading model using torch.load")
        
        # Check if weights are in Hugging Face format
        if any(k.startswith('model.') for k in state_dict.keys()):
            logger.info("Detected Hugging Face format weights, converting...")
            # Convert Hugging Face weight format to our format
            converted_state_dict = {}
            
            # Mapping rules
            mapping = {
                'model.embed_tokens.weight': 'embed_tokens.weight',
                'model.norm.weight': 'norm.weight',
                'lm_head.weight': 'lm_head.weight'
            }
            
            # Process each layer
            for i in range(self.config.num_layers):
                # Attention layer
                mapping.update({
                    f'model.layers.{i}.self_attn.q_proj.weight': f'layers.{i}.self_attn.q_proj.weight',
                    f'model.layers.{i}.self_attn.k_proj.weight': f'layers.{i}.self_attn.kv_proj_weight',  # Needs special handling
                    f'model.layers.{i}.self_attn.v_proj.weight': f'layers.{i}.self_attn.kv_proj_weight',  # Needs special handling
                    f'model.layers.{i}.self_attn.o_proj.weight': f'layers.{i}.self_attn.o_proj.weight',
                    f'model.layers.{i}.input_layernorm.weight': f'layers.{i}.attention_norm.weight',
                    
                    # MLP layer
                    f'model.layers.{i}.mlp.gate_proj.weight': f'layers.{i}.mlp.gate_proj.weight',
                    f'model.layers.{i}.mlp.up_proj.weight': f'layers.{i}.mlp.up_proj.weight',
                    f'model.layers.{i}.mlp.down_proj.weight': f'layers.{i}.mlp.down_proj.weight',
                    f'model.layers.{i}.post_attention_layernorm.weight': f'layers.{i}.ffn_norm.weight'
                })
                
                # If the model uses bias, add bias mapping
                if self.config.attention_bias:
                    mapping.update({
                        f'model.layers.{i}.self_attn.q_proj.bias': f'layers.{i}.self_attn.q_proj.bias',
                        f'model.layers.{i}.self_attn.o_proj.bias': f'layers.{i}.self_attn.o_proj.bias',
                    })
                
                if self.config.mlp_bias:
                    mapping.update({
                        f'model.layers.{i}.mlp.gate_proj.bias': f'layers.{i}.mlp.gate_proj.bias',
                        f'model.layers.{i}.mlp.up_proj.bias': f'layers.{i}.mlp.up_proj.bias',
                        f'model.layers.{i}.mlp.down_proj.bias': f'layers.{i}.mlp.down_proj.bias',
                    })
            
            # Process regular mappings
            for hf_name, our_name in mapping.items():
                if hf_name in state_dict:
                    converted_state_dict[our_name] = state_dict[hf_name]
            
            # Special handling for KV weights
            for i in range(self.config.num_layers):
                k_weight = state_dict.get(f'model.layers.{i}.self_attn.k_proj.weight')
                v_weight = state_dict.get(f'model.layers.{i}.self_attn.v_proj.weight')
                
                if k_weight is not None and v_weight is not None:
                    # Merge K and V weights
                    kv_weight = torch.cat([k_weight, v_weight], dim=0)
                    converted_state_dict[f'layers.{i}.self_attn.kv_proj_weight'] = kv_weight
            
            state_dict = converted_state_dict
            logger.info(f"Weight conversion completed, converted {len(state_dict)} parameters")
        
        # Load weights
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
            
        # Ensure model is in inference mode
        self.model.eval()
        logger.info("Model loading completed")
        logger.info(f"Model set to inference mode (model.training={self.model.training})")
    
    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Execute model forward pass (without gradient calculation)
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        Returns:
            logits: Model output logits
        """
        # Ensure model is in inference mode
        if self.model.training:
            logger.warning(f"Model unexpectedly in training mode, resetting to inference mode")
            self.model.eval()
            
        logger.debug(f"Executing forward pass (model.training={self.model.training})")
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        logger.debug(f"Executing forward inference: batch_size={batch_size}, seq_len={seq_len}")
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask
        )
    
    def generate(self, 
                 prompt: str, 
                 sampler: Sampler,
                 max_length: int = 100) -> str:
        """
        Generate text
        Args:
            prompt: Input prompt
            sampler: Sampler instance for token generation
            max_length: Maximum generation length
        Returns:
            Generated text
        """
        # Ensure model is in inference mode
        if self.model.training:
            logger.warning(f"Model unexpectedly in training mode, resetting to inference mode")
            self.model.eval()
            
        logger.info(f"Starting text generation, parameters: max_length={max_length}, sampler={type(sampler).__name__}")
        
        # Encode input text
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Record original input length, used to return only the generated part later
        input_length = input_ids.shape[1]
        logger.debug(f"Input prompt length: {input_length} tokens")
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # Generation loop
        for i in range(max_length):
            # Get model output
            logits = self.forward(input_ids, attention_mask)
            next_token_logits = logits[:, -1, :]
            
            # Use sampler to generate next token
            if isinstance(sampler, RepetitionPenaltySampler):
                # If using repetition penalty sampler, need to pass input_ids
                next_token = sampler.sample(next_token_logits, input_ids)
            else:
                next_token = sampler.sample(next_token_logits)
            
            # If EOS token is generated, stop generation
            if next_token.item() == self.config.eos_token_id:
                logger.debug(f"Generated EOS token at step {i+1}, stopping generation")
                break
                
            # Add new token to input sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            
            # Update sampler state
            sampler.update_state(input_ids)
        
        # Decode generated token sequence, return only the newly generated part
        generated_ids = input_ids[0, input_length:].tolist()
        generated_text = self.tokenizer.decode(generated_ids)
        logger.info(f"Generation completed, generated {len(generated_ids)} tokens")
        logger.debug(f"Generated text: {generated_text[:50]}..." if len(generated_text) > 50 else f"Generated text: {generated_text}")
        return generated_text