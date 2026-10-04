import numpy as np

def scaled_attention_weights(Q: np.ndarray, K: np.ndarray) -> list:
    """
    Compute scaled dot-product attention weights.

    Args:
        Q: (n_q, d_k) query matrix
        K: (n_k, d_k) key matrix

    Returns:
        Attention weights of shape (n_q, n_k) as a nested list,
        each entry rounded to 4 decimal places.
    """
    S = Q @ K.T / np.sqrt(K.shape[-1])
    S = S - np.max(S, axis=-1, keepdims = True)
    return np.exp(S) / np.sum(np.exp(S), axis=-1, keepdims = True)
    pass
