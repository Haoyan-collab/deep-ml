import torch

def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """
    Apply Rotary Positional Embeddings (RoPE) to input embeddings.
    
    Args:
        x: Input embeddings of shape (seq_len, d), d must be even
        positions: Position indices of shape (seq_len,)
        base: Base for frequency computation (default: 10000.0)
    
    Returns:
        Embeddings with rotary positional encoding applied, shape (seq_len, d)
    """
    # Your code here
    d_model = x.shape[-1]
    dim = torch.arange(0,d_model,2,device = x.device)

    theta = 1 / (base ** (dim/d_model)) #(d_model//2,)
    angles = positions[:,None] * theta

    sin = torch.sin(angles)
    cos = torch.cos(angles)

    pe = torch.zeros((positions.shape[0],d_model))
    pe[:,0::2] = x[:,0::2] * cos - x[:,1::2] * sin
    pe[:,1::2] = x[:,0::2] * sin + x[:,1::2] * cos

    return pe