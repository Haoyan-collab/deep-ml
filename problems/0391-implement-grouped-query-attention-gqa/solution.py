import torch
import torch.nn.functional as F
import numpy as np
def grouped_query_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, num_heads: int, num_kv_heads: int) -> torch.Tensor:
    """
    Compute Grouped Query Attention.

    Args:
        Q: Query tensor, shape (batch_size, seq_len, num_heads * head_dim)
        K: Key tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        V: Value tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads

    Returns:
        Output tensor, shape (batch_size, seq_len, num_heads * head_dim)
    """
    seq_len = Q.shape[-2]
    B = Q.shape[0]
    d_q = Q.shape[-1]
    Q = Q.reshape(B, seq_len, num_heads, d_q//num_heads).transpose(1,2)
    K = K.reshape(B, seq_len, num_kv_heads, K.shape[-1]//num_kv_heads).transpose(1,2)
    V = V.reshape(B, seq_len, num_kv_heads, V.shape[-1]//num_kv_heads).transpose(1,2)

    #iterate
    K = K.repeat_interleave(num_heads // num_kv_heads, dim = 1)
    V = V.repeat_interleave(num_heads // num_kv_heads, dim = 1)

    score = Q @ K.transpose(-1,-2) / np.sqrt(K.shape[-1])
    score = torch.softmax(score, dim=-1)

    out = score @ V
    return out.transpose(1,2).reshape(B, seq_len, d_q)





