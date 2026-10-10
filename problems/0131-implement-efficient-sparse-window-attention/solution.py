import torch
import numpy as np
def sparse_window_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, window_size: int, scale_factor: float = None) -> torch.Tensor:
    """Computes sparse attention with a sliding window mask.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        window_size: The radius of the attention window
        scale_factor: Scaling factor for the dot product. If None, uses sqrt(d_k).
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    seq_len = Q.shape[-2]
    d_k = Q.shape[-1]
    out = torch.zeros((seq_len,V.shape[-1]),device= V.device)
    if scale_factor is None:
        scale_factor = np.sqrt(d_k)
    for i in range(seq_len):
        left = max(0, i - window_size)
        right = min(seq_len - 1, i + window_size)
        q = Q[i:i+1, :]
        k = K[left : right + 1, :]
        v = V[left : right + 1, :]

        score = q @ k.transpose(-1,-2) / scale_factor
        score = torch.softmax(score, dim=-1)
        out[i:i+1,:] = score @ v

    return out