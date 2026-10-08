import numpy as np

def apply_rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
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
    theta = 1 / base ** (2 * np.arange(0, d_model//2, 1) / d_model)
    angels = positions[:,None] * theta[None,:]

    x_even = x[:,0::2]
    x_odd = x[:,1::2]

    cos = np.cos(angels)
    sin = np.sin(angels)

    out = np.zeros(x.shape)
    out[:,0::2] = x_even * cos - x_odd * sin
    out[:,1::2] = x_even * sin + x_odd * cos

    return out