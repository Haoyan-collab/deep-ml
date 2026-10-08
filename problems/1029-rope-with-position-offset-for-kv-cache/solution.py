import numpy as np

def apply_rope_with_offset(x, cos, sin, start_pos):
    """
    Apply rotary positional embedding to x starting at absolute position start_pos.

    Args:
        x: array-like of shape (batch, n_heads, seq_len, head_dim)
        cos: array-like of shape (max_seq_len, head_dim)
        sin: array-like of shape (max_seq_len, head_dim)
        start_pos: int, absolute position of the first token in x

    Returns:
        Nested list with the same shape as x.
    """
    x = np.array(x)
    cos = np.array(cos)
    sin = np.array(sin)
    seq_len = x.shape[-2]
    cos = cos[start_pos:start_pos + seq_len]
    sin = sin[start_pos:start_pos + seq_len]

    head_dim = x.shape[-1]
    x1 = x[:,:,:,:head_dim//2]
    x2 = x[:,:,:,head_dim//2:]
    rotated = np.concatenate([-x2,x1], axis=-1)

    out = x * cos + rotated * sin
    return out.tolist()
