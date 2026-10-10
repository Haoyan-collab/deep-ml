import torch
import numpy as np
def multiquery_attention(X: torch.Tensor, W_queries: list, W_key: torch.Tensor, W_value: torch.Tensor, W_out: torch.Tensor) -> torch.Tensor:
    """
    Compute Multi-Query Attention.
    
    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_queries: List of query weight tensors, each (d_model, d_k), one per head
        W_key: Shared key weight tensor of shape (d_model, d_k)
        W_value: Shared value weight tensor of shape (d_model, d_v)
        W_out: Output projection tensor of shape (num_heads * d_v, d_model)
    
    Returns:
        Output tensor of shape (seq_len, d_model), rounded to 4 decimal places
    """
    out = []
    K = X @ W_key
    V = X @ W_value

    for W_q in W_queries:
        Q = X @ W_q
        score = Q @ K.transpose(-1,-2) / np.sqrt(K.shape[-1])
        score = torch.softmax(score, dim=-1)
        out.append(score @ V)

    out = torch.cat(out, dim=-1)
    out = out @ W_out
    return torch.round(out, decimals = 4)