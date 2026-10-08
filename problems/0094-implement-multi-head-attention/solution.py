import torch
import torch.nn.functional as F
from typing import Tuple
import numpy as np
def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return tuple([Q,K,V])

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    d_k = Q.shape[-1]
    score = Q @ K.transpose(-1,-2) / np.sqrt(d_k)
    score -= torch.max(score, dim=-1,keepdim = True).values
    sft_score = torch.exp(score) / torch.sum(torch.exp(score), dim = -1, keepdim = True)
    return sft_score @ V

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    seq_len = Q.shape[-2]
    d_model = Q.shape[-1]
    assert d_model % n_heads == 0, "not divisible"
    d_k = d_model // n_heads
    Q = Q.reshape(seq_len,n_heads,d_k).transpose(0,1)
    K = K.reshape(seq_len,n_heads,d_k).transpose(0,1)
    V = V.reshape(seq_len,n_heads,d_k).transpose(0,1)

    out = self_attention(Q,K,V)
    out = out.transpose(0,1).reshape(seq_len,d_model)

    return out
















