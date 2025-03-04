import torch
import torch.nn.functional as F
import math

def flash_attention2_no_pad(q, k, v, qk_scale, b_start_loc, b_seq_len, seq_len):
    """
    Implement basic Flash Attention 2.0 algorithm (no padding version)
    Args:
        q: [batch_size*seq_len, num_heads, head_dim]
        k: [batch_size*seq_len, num_kv_heads, head_dim]
        v: [batch_size*seq_len, num_kv_heads, head_dim]
        qk_scale: scaling factor
        b_start_loc: starting position for each batch
        b_seq_len: sequence length for each batch
        seq_len: total sequence length
    Returns:
        output: [batch_size*seq_len, num_heads, head_dim]
    """
    batch_size = len(b_seq_len)
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    
    # Reshape input tensors
    q = q.view(batch_size, seq_len, num_heads, head_dim)
    k = k.view(batch_size, seq_len, k.shape[1], head_dim)
    v = v.view(batch_size, seq_len, v.shape[1], head_dim)
    
    # Calculate attention scores
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * qk_scale
    
    # Apply softmax
    attn_weights = F.softmax(attn_weights, dim=-1)
    
    # Calculate output
    output = torch.matmul(attn_weights, v)
    
    return output.view(-1, num_heads, head_dim)

def flash_decoding(q, k, v, qk_scale, b_req_tokens_table, b_seq_len, max_actual_seq_len):
    """
    Implement Flash Attention for decoding phase
    Args:
        q: [batch_size, num_heads, head_dim]
        k: [batch_size, num_kv_heads, head_dim]
        v: [batch_size, num_kv_heads, head_dim]
        qk_scale: scaling factor
        b_req_tokens_table: requested token table
        b_seq_len: sequence length for each batch
        max_actual_seq_len: actual maximum sequence length
    Returns:
        output: [batch_size, num_heads, head_dim]
    """
    # Calculate attention scores
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * qk_scale
    
    # Apply softmax
    attn_weights = F.softmax(attn_weights, dim=-1)
    
    # Calculate output
    output = torch.matmul(attn_weights, v)
    
    return output

def rope_emb_forward(q, k, cos, sin, batch_size, seq_len):
    """
    Implement Rotary Position Encoding
    Args:
        q: [batch_size*seq_len, num_heads, head_dim] or [batch_size, num_heads, head_dim]
        k: [batch_size*seq_len, num_kv_heads, head_dim] or [batch_size, num_kv_heads, head_dim]
        cos: [batch_size, seq_len, head_dim]
        sin: [batch_size, seq_len, head_dim]
    Returns:
        q_out: q after applying RoPE
        k_out: k after applying RoPE
    """
    # Split q and k into real and imaginary parts
    q_real, q_imag = q[..., ::2], q[..., 1::2]
    k_real, k_imag = k[..., ::2], k[..., 1::2]
    
    # Apply rotation
    q_out_real = q_real * cos - q_imag * sin
    q_out_imag = q_real * sin + q_imag * cos
    k_out_real = k_real * cos - k_imag * sin
    k_out_imag = k_real * sin + k_imag * cos
    
    # Combine real and imaginary parts
    q_out = torch.stack([q_out_real, q_out_imag], dim=-1).flatten(-2)
    k_out = torch.stack([k_out_real, k_out_imag], dim=-1).flatten(-2)
    
    return q_out, k_out

def update_kv_buffer(combined_kv, cur_select_index, kv_buffer):
    """
    Update KV cache
    Args:
        combined_kv: new KV values
        cur_select_index: current selected index
        kv_buffer: KV cache
    """
    kv_buffer[cur_select_index] = combined_kv

def skip_rmsnorm(hidden_states, residual, weight, eps):
    """
    Implement RMSNorm with skip connection
    Args:
        hidden_states: input states
        residual: residual connection
        weight: weights
        eps: epsilon value
    Returns:
        normalized_hidden_states: normalized hidden states
        residual: updated residual
    """
    if residual is None:
        residual = hidden_states
    else:
        hidden_states = hidden_states + residual
    
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states, hidden_states

def swiglu_forward(gate_output, up_output):
    """
    Implement SwiGLU activation function
    Args:
        gate_output: gate output
        up_output: up-projection output
    Returns:
        output: SwiGLU output
    """
    return F.silu(gate_output) * up_output 