import torch
import torch.nn.functional as F
import os
import logging
from typing import Optional, Dict, Any, List, Union
from .base_engine import BaseEngine
from ..model.llama import LlamaModel
from ..model.model_config import LlamaConfig
from ..tokenizer import Tokenizer

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
        logger.info(f"Model set to inference mode (model.training={self.model.training}) [llm_engine.py:137]")
    
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
            logger.warning(f"Model unexpectedly in training mode, resetting to inference mode [llm_engine.py:150]")
            self.model.eval()
            
        logger.debug(f"Executing forward pass (model.training={self.model.training}) [llm_engine.py:153]")
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
                 max_length: int = 100, 
                 temperature: float = 0.7, 
                 top_p: float = 0.9,
                 top_k: int = 0,
                 repetition_penalty: float = 1.0,
                 do_sample: bool = True) -> str:
        """
        Generate text
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Temperature parameter
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty parameter
            do_sample: Whether to use sampling (False for greedy decoding)
        Returns:
            Generated text
        """
        # Ensure model is in inference mode
        if self.model.training:
            logger.warning(f"Model unexpectedly in training mode, resetting to inference mode [llm_engine.py:180]")
            self.model.eval()
            
        logger.info(f"Starting text generation, parameters: max_length={max_length}, temperature={temperature}, top_p={top_p}, top_k={top_k}, repetition_penalty={repetition_penalty}, do_sample={do_sample}")
        
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
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    if next_token_logits[0, token_id] < 0:
                        next_token_logits[0, token_id] *= repetition_penalty
                    else:
                        next_token_logits[0, token_id] /= repetition_penalty
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # If using sampling
            if do_sample:
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = torch.topk(next_token_logits, k=top_k)[0][:, -1, None]
                    next_token_logits[next_token_logits < indices_to_remove] = float('-inf')
                
                # Apply top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # If EOS token is generated, stop generation
            if next_token.item() == self.config.eos_token_id:
                logger.debug(f"Generated EOS token at step {i+1}, stopping generation")
                break
                
            # Add new token to input sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
        
        # Decode generated token sequence, return only the newly generated part
        generated_ids = input_ids[0, input_length:].tolist()
        generated_text = self.tokenizer.decode(generated_ids)
        logger.info(f"Generation completed, generated {len(generated_ids)} tokens")
        logger.debug(f"Generated text: {generated_text[:50]}..." if len(generated_text) > 50 else f"Generated text: {generated_text}")
        return generated_text