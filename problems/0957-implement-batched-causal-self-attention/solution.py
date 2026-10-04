import numpy as np

def batched_causal_self_attention(X, W_query, W_key, W_value):
    """
    Batched causal self-attention.

    Args:
        X: nested list of shape (B, T, d_in)
        W_query, W_key, W_value: nested lists of shape (d_in, d_out)

    Returns:
        Nested list of shape (B, T, d_out) -- the context vectors.
    """
    # Your code here
    X = np.array(X)
    W_query = np.array(W_query)
    W_key = np.array(W_key)
    W_value = np.array(W_value)
    Q = X @ W_query
    K = X @ W_key
    V = X @ W_value
    S = Q @ K.swapaxes(-1,-2) / np.sqrt(K.shape[-1])

    q_len = Q.shape[-2]
    k_len = K.shape[-2]
    q_idx = np.arange(q_len)[:,None]
    k_idx = np.arange(k_len)[None,:]
    mask = k_idx > q_idx

    S = np.where(mask, -np.inf, S)
    S = S - np.max(S, axis=-1, keepdims = True)
    sft_s = np.exp(S) / np.sum(np.exp(S), axis=-1, keepdims = True)
    return (sft_s @ V).tolist()
    pass
