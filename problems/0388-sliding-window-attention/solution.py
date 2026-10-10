import torch
import math
def sliding_window_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Compute sliding window attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        window_size: Number of positions to the left and right each query can attend to
    
    Returns:
        Output matrix of shape (seq_len, d_v), rounded to 4 decimal places.
    """
    d_model = K.shape[-1]
    score = Q @ K.transpose(-1,-2) / math.sqrt(d_model)

    #construct mask
    q_len = Q.shape[-2]
    pos = torch.arange(q_len, device = Q.device)
    distance = torch.abs(pos[:,None] - pos[None, :])
    mask = distance <= window_size

    score = torch.where(mask, score, -torch.inf)
    score = torch.softmax(score, dim=-1)

    out = score @ V
    return torch.round(out, decimals = 4)