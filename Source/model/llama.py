import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from typing import Optional, Tuple, Dict, Any
from ..kernel.rotary_embedding import RotaryEmbedding
from .model_config import LlamaConfig

# Create module-level logger
logger = logging.getLogger(__name__)

def get_activation_fn(activation_name: str):
    """Get activation function"""
    if activation_name == "silu":
        return F.silu
    elif activation_name == "gelu":
        return F.gelu
    elif activation_name == "relu":
        return F.relu
    else:
        logger.error(f"Unsupported activation function: {activation_name}")
        raise ValueError(f"Unsupported activation function: {activation_name}")

class FusedAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        # K V heads are the same, but may differ from Q
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        
        self.num_q_heads = config.num_heads
        self.hidden_size = config.hidden_size
        
        # Decide whether to use bias based on configuration
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.kv_proj_weight = nn.Parameter(torch.empty(self.num_kv_heads * self.head_dim * 2, self.hidden_size))
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        
        # Use default initialization
        self._init_weights()
        logger.debug(f"Initializing FusedAttention: num_q_heads={self.num_q_heads}, num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}")

    def _init_weights(self):
        """Use standard initialization method"""
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.kv_proj_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=0.02)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)

    def forward(self, x: torch.Tensor, position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        
        # Calculate Q K V
        q = self.q_proj(x)
        k_proj_weight, v_proj_weight = torch.split(self.kv_proj_weight, self.num_kv_heads * self.head_dim, dim=0)
        k = F.linear(x, k_proj_weight)
        v = F.linear(x, v_proj_weight)
        
        # Reshape
        q = q.view(batch_size, seq_len, self.num_q_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply rotary position encoding
        if position_embeddings is not None:
            cos, sin = position_embeddings
            
            # Print detailed shape information for debugging
            logger.debug(f"Position encoding shape: cos.shape={cos.shape}, sin.shape={sin.shape}")
            logger.debug(f"Query key value shape: q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")
            
            # Ensure cos and sin shapes are compatible with q and k
            # Check the third dimension (sequence length)
            if cos.size(2) != seq_len:
                logger.warning(f"Position encoding sequence length mismatch: cos.size(2)={cos.size(2)}, seq_len={seq_len}")
                # If position encoding sequence length is greater than input sequence length, truncate
                if cos.size(2) > seq_len:
                    cos = cos[:, :, :seq_len, :]
                    sin = sin[:, :, :seq_len, :]
                # If position encoding sequence length is less than input sequence length, extend
                else:
                    # Record original length
                    orig_len = cos.size(2)
                    # Create new position encoding tensors, initialized to 0
                    new_cos = torch.zeros(cos.size(0), cos.size(1), seq_len, cos.size(3), device=cos.device, dtype=cos.dtype)
                    new_sin = torch.zeros(sin.size(0), sin.size(1), seq_len, sin.size(3), device=sin.device, dtype=sin.dtype)
                    # Copy original position encodings
                    new_cos[:, :, :orig_len, :] = cos
                    new_sin[:, :, :orig_len, :] = sin
                    # For positions beyond the original length, use the last position encoding
                    if orig_len > 0:
                        new_cos[:, :, orig_len:, :] = cos[:, :, -1:, :]
                        new_sin[:, :, orig_len:, :] = sin[:, :, -1:, :]
                    cos = new_cos
                    sin = new_sin
                logger.debug(f"Position encoding shape after sequence length adjustment: cos.shape={cos.shape}, sin.shape={sin.shape}")
            
            # Ensure batch size matches
            if cos.size(0) != batch_size and cos.size(0) == 1:
                cos = cos.expand(batch_size, -1, -1, -1)
                sin = sin.expand(batch_size, -1, -1, -1)
                logger.debug(f"Position encoding shape after batch size expansion: cos.shape={cos.shape}, sin.shape={sin.shape}")
            
            # Ensure number of heads matches
            # For q, ensure that the 1st dimension of cos and sin matches num_q_heads
            if cos.size(1) != self.num_q_heads:
                logger.warning(f"Position encoding head count mismatch: cos.size(1)={cos.size(1)}, num_q_heads={self.num_q_heads}, adjusting")
                # If position encoding head count is 1, expand to required number
                if cos.size(1) == 1:
                    cos = cos.expand(-1, self.num_q_heads, -1, -1)
                    sin = sin.expand(-1, self.num_q_heads, -1, -1)
                # Otherwise, truncate or repeat
                else:
                    # Create new position encoding tensors
                    new_cos = torch.zeros(cos.size(0), self.num_q_heads, cos.size(2), cos.size(3), device=cos.device, dtype=cos.dtype)
                    new_sin = torch.zeros(sin.size(0), self.num_q_heads, sin.size(2), sin.size(3), device=sin.device, dtype=sin.dtype)
                    # Copy position encodings
                    for i in range(self.num_q_heads):
                        src_idx = i % cos.size(1)
                        new_cos[:, i, :, :] = cos[:, src_idx, :, :]
                        new_sin[:, i, :, :] = sin[:, src_idx, :, :]
                    cos = new_cos
                    sin = new_sin
                logger.debug(f"Position encoding shape after head count adjustment: cos.shape={cos.shape}, sin.shape={sin.shape}")
            
            # Ensure the last dimension (head_dim) matches
            if cos.size(3) != self.head_dim:
                logger.warning(f"Position encoding dimension mismatch: cos.size(3)={cos.size(3)}, head_dim={self.head_dim}, adjusting")
                # If position encoding dimension is greater than head_dim, truncate
                if cos.size(3) > self.head_dim:
                    cos = cos[:, :, :, :self.head_dim]
                    sin = sin[:, :, :, :self.head_dim]
                # If position encoding dimension is less than head_dim, pad
                else:
                    # Create new position encoding tensors
                    new_cos = torch.zeros(cos.size(0), cos.size(1), cos.size(2), self.head_dim, device=cos.device, dtype=cos.dtype)
                    new_sin = torch.zeros(sin.size(0), sin.size(1), sin.size(2), self.head_dim, device=sin.device, dtype=sin.dtype)
                    # Copy position encodings
                    new_cos[:, :, :, :cos.size(3)] = cos
                    new_sin[:, :, :, :sin.size(3)] = sin
                    # For dimensions beyond the original, use 0
                    cos = new_cos
                    sin = new_sin
                logger.debug(f"Position encoding shape after dimension adjustment: cos.shape={cos.shape}, sin.shape={sin.shape}")
            
            # Final check if shapes match
            if cos.size(0) != q.size(0) or cos.size(1) != q.size(2) or cos.size(2) != q.size(1) or cos.size(3) != q.size(3):
                logger.error(f"Position encoding shape does not match query shape: cos.shape={cos.shape}, q.shape={q.shape}")
                # Adjust cos and sin shapes to match q
                # Note: We need to transpose cos and sin because q's shape is [batch_size, seq_len, num_heads, head_dim]
                # while cos and sin's shape is [batch_size, num_heads, seq_len, head_dim]
                cos_adjusted = cos.permute(0, 2, 1, 3)
                sin_adjusted = sin.permute(0, 2, 1, 3)
                logger.warning(f"Adjusted position encoding shape: cos_adjusted.shape={cos_adjusted.shape}, sin_adjusted.shape={sin_adjusted.shape}")
                
                # Apply rotary position encoding
                q_embed = (q * cos_adjusted) + (self._rotate_half(q) * sin_adjusted)
                k_embed = (k * cos_adjusted[:, :, :self.num_kv_heads, :]) + (self._rotate_half(k) * sin_adjusted[:, :, :self.num_kv_heads, :])
            else:
                # Apply rotary position encoding
                # Transpose cos and sin to match q and k shapes
                cos_adjusted = cos.permute(0, 2, 1, 3)
                sin_adjusted = sin.permute(0, 2, 1, 3)
                q_embed = (q * cos_adjusted) + (self._rotate_half(q) * sin_adjusted)
                k_embed = (k * cos_adjusted[:, :, :self.num_kv_heads, :]) + (self._rotate_half(k) * sin_adjusted[:, :, :self.num_kv_heads, :])
            
            q, k = q_embed, k_embed
        
        # Transpose for attention calculation
        q = q.transpose(1, 2)  # [batch_size, num_q_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_kv_heads, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_kv_heads, seq_len, head_dim]
        
        # Calculate attention scores
        scale = 1.0 / (self.head_dim ** 0.5)
        
        # Check if the number of heads in q and k match, adjust if not
        if q.size(1) != k.size(1):
            logger.debug(f"Number of heads in q and k don't match: q.shape={q.shape}, k.shape={k.shape}")
            
            # If q's head count is a multiple of k's (grouped query attention mechanism)
            if q.size(1) % k.size(1) == 0:
                # Calculate number of Q heads per KV head
                num_q_per_kv = q.size(1) // k.size(1)
                logger.debug(f"Using grouped query attention: {num_q_per_kv} Q heads per KV head")
                
                # Reshape q to match k's head structure
                q_reshaped = q.view(batch_size, k.size(1), num_q_per_kv, seq_len, self.head_dim)
                
                # Process each group
                outputs = []
                for i in range(num_q_per_kv):
                    q_group = q_reshaped[:, :, i]  # [batch_size, num_kv_heads, seq_len, head_dim]
                    scores_group = torch.matmul(q_group, k.transpose(-2, -1)) * scale
                    
                    # Apply attention mask (if provided)
                    if attention_mask is not None:
                        scores_group = scores_group + attention_mask
                    
                    # Apply softmax
                    attention_weights_group = F.softmax(scores_group, dim=-1)
                    
                    # Calculate output
                    output_group = torch.matmul(attention_weights_group, v)  # [batch_size, num_kv_heads, seq_len, head_dim]
                    outputs.append(output_group)
                
                # Merge all group outputs
                output = torch.cat([out.unsqueeze(2) for out in outputs], dim=2)
                output = output.view(batch_size, q.size(1), seq_len, self.head_dim)
            else:
                # If not an integer multiple relationship, emit warning and try to copy k and v to match q's head count
                logger.error(f"q's head count ({q.size(1)}) is not an integer multiple of k's ({k.size(1)}), trying to copy k and v")
                repeat_factor = q.size(1) // k.size(1)
                remainder = q.size(1) % k.size(1)
                
                if remainder == 0:
                    # Simple copy
                    k_expanded = k.repeat(1, repeat_factor, 1, 1)
                    v_expanded = v.repeat(1, repeat_factor, 1, 1)
                else:
                    # Copy and add extra heads
                    k_expanded = torch.cat([k.repeat(1, repeat_factor, 1, 1), k[:, :remainder]], dim=1)
                    v_expanded = torch.cat([v.repeat(1, repeat_factor, 1, 1), v[:, :remainder]], dim=1)
                
                logger.debug(f"Expanded k and v shapes: k_expanded.shape={k_expanded.shape}, v_expanded.shape={v_expanded.shape}")
                
                # Use expanded k and v to calculate attention
                scores = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale
                
                # Apply attention mask (if provided)
                if attention_mask is not None:
                    scores = scores + attention_mask
                
                # Apply softmax
                attention_weights = F.softmax(scores, dim=-1)
                
                # Calculate output
                output = torch.matmul(attention_weights, v_expanded)
        else:
            # Head count matches, normal attention calculation
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            # Apply attention mask (if provided)
            if attention_mask is not None:
                scores = scores + attention_mask
            
            # Apply softmax
            attention_weights = F.softmax(scores, dim=-1)
            
            # Calculate output
            output = torch.matmul(attention_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Transpose and reshape output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Apply output projection
        return self.o_proj(output)
    
    def _rotate_half(self, x):
        """Helper function to implement rotation operation"""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

class FusedMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.activation_fn = get_activation_fn(config.hidden_act)

        # Decide whether to use bias based on configuration
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        
        # Use default initialization
        self._init_weights()
        logger.debug(f"Initializing FusedMLP: hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}")

    def _init_weights(self):
        """Use standard initialization method"""
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)
        if self.gate_proj.bias is not None:
            nn.init.zeros_(self.gate_proj.bias)
        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)

    def forward(self, x):
        # Apply activation function
        return self.down_proj(self.activation_fn(self.gate_proj(x)) * self.up_proj(x))

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        
    def forward(self, x):
        # Calculate RMS
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        
        # Attention layer
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = FusedAttention(config)
        
        # Feedforward network layer
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FusedMLP(config)

    def forward(self, x, position_embeddings=None, attention_mask=None):
        # Apply attention layer
        residual = x
        x = self.attention_norm(x)
        x = self.self_attn(x, position_embeddings, attention_mask)
        x = residual + x
        
        # Apply feedforward network layer
        residual = x
        x = self.ffn_norm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x

class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)
        
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(config.num_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Decide whether to bind weights based on configuration
        if config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.embed_tokens.weight
            logger.info("Using weight binding: lm_head weights share with embed_tokens weights")
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            logger.info("Not using weight binding: lm_head weights initialized independently")
        
        # Initialize parameters
        self._init_weights()
        logger.info(f"Initializing LlamaModel: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}, num_layers={config.num_layers}")
    
    def _init_weights(self):
        """Use standard initialization method"""
        # Initialize embedding layer
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        logger.debug("Initializing embed_tokens weights")
        
        # Initialize LM head (if not shared weights)
        if not self.config.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
            logger.debug("Initializing lm_head weights")
    
    def forward(self, input_ids, position_ids=None, attention_mask=None, inputs_embeds=None):
        # Ensure model is in inference mode
        if self.training:
            logger.warning(f"Model is currently in training mode, but this is an inference framework. Automatically switching to inference mode")
            self.eval()
        
        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        hidden_states = inputs_embeds
        batch_size, seq_len = hidden_states.shape[:2]
        logger.debug(f"Model forward propagation: batch_size={batch_size}, seq_len={seq_len}")
        
        # Prepare position encodings
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        
        # Get rotary position encodings
        try:
            # Get rotary position encodings
            cos, sin = self.rotary_emb.forward(hidden_states, seq_len)
            logger.debug(f"Getting rotary position encodings: cos.shape={cos.shape}, sin.shape={sin.shape}")
            
            # Ensure position encoding sequence length matches input sequence length
            if cos.size(2) != seq_len:
                logger.warning(f"Position encoding sequence length mismatch: cos.size(2)={cos.size(2)}, seq_len={seq_len}, adjusting")
                # If position encoding sequence length is greater than input sequence length, truncate
                if cos.size(2) > seq_len:
                    cos = cos[:, :, :seq_len, :]
                    sin = sin[:, :, :seq_len, :]
                # If position encoding sequence length is less than input sequence length, extend
                else:
                    # Record original length
                    orig_len = cos.size(2)
                    # Create new position encoding tensors, initialized to 0
                    new_cos = torch.zeros(cos.size(0), cos.size(1), seq_len, cos.size(3), device=cos.device, dtype=cos.dtype)
                    new_sin = torch.zeros(sin.size(0), sin.size(1), seq_len, sin.size(3), device=sin.device, dtype=sin.dtype)
                    # Copy original position encodings
                    new_cos[:, :, :orig_len, :] = cos
                    new_sin[:, :, :orig_len, :] = sin
                    # For positions beyond the original length, use the last position encoding
                    if orig_len > 0:
                        new_cos[:, :, orig_len:, :] = cos[:, :, -1:, :]
                        new_sin[:, :, orig_len:, :] = sin[:, :, -1:, :]
                    cos = new_cos
                    sin = new_sin
                logger.debug(f"Position encoding shape after sequence length adjustment: cos.shape={cos.shape}, sin.shape={sin.shape}")
            
            # Ensure position encoding head count matches model head count
            num_heads = self.config.num_heads
            if cos.size(1) != num_heads:
                logger.warning(f"Position encoding head count mismatch: cos.size(1)={cos.size(1)}, num_heads={num_heads}, adjusting")
                # If position encoding head count is 1, expand to required number
                if cos.size(1) == 1:
                    cos = cos.expand(-1, num_heads, -1, -1)
                    sin = sin.expand(-1, num_heads, -1, -1)
                # Otherwise, truncate or repeat
                else:
                    # Create new position encoding tensors
                    new_cos = torch.zeros(cos.size(0), num_heads, cos.size(2), cos.size(3), device=cos.device, dtype=cos.dtype)
                    new_sin = torch.zeros(sin.size(0), num_heads, sin.size(2), sin.size(3), device=sin.device, dtype=sin.dtype)
                    # Copy position encodings
                    for i in range(num_heads):
                        src_idx = i % cos.size(1)
                        new_cos[:, i, :, :] = cos[:, src_idx, :, :]
                        new_sin[:, i, :, :] = sin[:, src_idx, :, :]
                    cos = new_cos
                    sin = new_sin
                logger.debug(f"Position encoding shape after head count adjustment: cos.shape={cos.shape}, sin.shape={sin.shape}")
            
            # Ensure position encoding dimension matches model head dimension
            head_dim = self.config.head_dim
            if cos.size(3) != head_dim:
                logger.warning(f"Position encoding dimension mismatch: cos.size(3)={cos.size(3)}, head_dim={head_dim}, adjusting")
                # If position encoding dimension is greater than head_dim, truncate
                if cos.size(3) > head_dim:
                    cos = cos[:, :, :, :head_dim]
                    sin = sin[:, :, :, :head_dim]
                # If position encoding dimension is less than head_dim, pad
                else:
                    # Create new position encoding tensors
                    new_cos = torch.zeros(cos.size(0), cos.size(1), cos.size(2), head_dim, device=cos.device, dtype=cos.dtype)
                    new_sin = torch.zeros(sin.size(0), sin.size(1), sin.size(2), head_dim, device=sin.device, dtype=sin.dtype)
                    # Copy position encodings
                    new_cos[:, :, :, :cos.size(3)] = cos
                    new_sin[:, :, :, :sin.size(3)] = sin
                    # For dimensions beyond the original, use 0
                    cos = new_cos
                    sin = new_sin
                logger.debug(f"Position encoding shape after dimension adjustment: cos.shape={cos.shape}, sin.shape={sin.shape}")
            
            position_embeddings = (cos, sin)
            logger.debug(f"Final position encoding shape: cos.shape={cos.shape}, sin.shape={sin.shape}")
        except Exception as e:
            logger.error(f"Error generating position encodings: {e}")
            # If error, create a zero tensor of the same shape as the model configuration as position encodings
            device = hidden_states.device
            dtype = hidden_states.dtype
            
            # Create all-zero position encodings
            cos = torch.ones((1, self.config.num_heads, seq_len, self.config.head_dim), device=device, dtype=dtype)
            sin = torch.zeros((1, self.config.num_heads, seq_len, self.config.head_dim), device=device, dtype=dtype)
            
            position_embeddings = (cos, sin)
            logger.warning(f"Using emergency position encodings: cos.shape={cos.shape}, sin.shape={sin.shape}")
        
        # Prepare attention mask
        if attention_mask is not None:
            # Ensure mask shape is correct [batch_size, 1, 1, seq_len]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Convert mask to additive mask (-inf for masked positions)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
            logger.debug(f"Applying attention mask: shape={attention_mask.shape}")
        
        # Apply all layers
        for i, layer in enumerate(self.layers):
            logger.debug(f"Processing layer {i+1}/{len(self.layers)}, inference mode: {not self.training}")
            hidden_states = layer(hidden_states, position_embeddings, attention_mask)
            if i % 8 == 0:  # Record every 8 layers, to avoid too much logging
                logger.debug(f"Completed processing layer {i+1}/{len(self.layers)}")
        
        # Apply final layer normalization
        hidden_states = self.norm(hidden_states)
        
        # Calculate language model head
        logits = self.lm_head(hidden_states)
        logger.debug(f"Generating logits: shape={logits.shape}, inference mode: {not self.training}")
        
        return logits