import numpy as np

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    q = x_new @ W_Q
    k = x_new @ W_K
    v = x_new @ W_V
    (k_cache, v_cache) = cache if cache is not None else (None, None)
    if k_cache is not None:
        k_cache = np.vstack([k_cache,k])
        v_cache = np.vstack([v_cache,v])
    else:
        k_cache = k[None,:]
        v_cache = v[None,:]
    S = q @ k_cache.swapaxes(-1,-2) / np.sqrt(k_cache.shape[-1])
    S = S - np.max(S, axis=-1, keepdims = True)
    sft_S = np.exp(S) / np.sum(np.exp(S), axis=-1, keepdims = True)
    
    return (sft_S @ v_cache, (k_cache,v_cache))
    
    pass