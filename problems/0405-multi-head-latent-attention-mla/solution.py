import torch
import math
def multi_head_latent_attention(
    X: torch.Tensor,
    W_dkv: torch.Tensor,
    W_uk: torch.Tensor,
    W_uv: torch.Tensor,
    W_dq: torch.Tensor,
    W_uq: torch.Tensor,
    W_o: torch.Tensor,
    n_heads: int
) -> tuple:
    """
    Perform Multi-Head Latent Attention (MLA).

    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_dkv: Down-projection for KV compression (d_model, d_c_kv)
        W_uk: Up-projection for keys (d_c_kv, d_model)
        W_uv: Up-projection for values (d_c_kv, d_model)
        W_dq: Down-projection for query compression (d_model, d_c_q)
        W_uq: Up-projection for queries (d_c_q, d_model)
        W_o: Output projection (d_model, d_model)
        n_heads: Number of attention heads

    Returns:
        Tuple of (output, c_kv) where:
        - output: shape (seq_len, d_model)
        - c_kv: compressed KV latent of shape (seq_len, d_c_kv)
    """
    # downsample and upsample k,v
    c_kv = X @ W_dkv

    k = c_kv @ W_uk
    v = c_kv @ W_uv
    
    # down and up q
    c_q = X @ W_dq
    q = c_q @ W_uq

    # multi_head_attention
    seq_len = q.shape[-2]
    d_model = q.shape[-1]
    
    # split heads
    assert d_model % n_heads == 0, "not divisible"
    d_k = d_model // n_heads
    q = q.reshape(seq_len,n_heads,d_k).transpose(0,1)
    k = k.reshape(seq_len,n_heads,d_k).transpose(0,1)
    v = v.reshape(seq_len,n_heads,d_k).transpose(0,1)

    score = q @ k.transpose(-1,-2) / math.sqrt(d_k)
    attn = torch.softmax(score, dim=-1)
    out = attn @ v

    # merge heads
    out = out.transpose(0,1).reshape(seq_len,d_model)
    out = out @ W_o

    return (out, c_kv)



