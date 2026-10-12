import torch
import math

def kv_cache_attention_step(x_new: torch.Tensor, W_Q: torch.Tensor, W_K: torch.Tensor, W_V: torch.Tensor, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) of tensors or None if first step
    
    Returns:
        Tuple (output, updated_cache) where output is shape (d_v,)
        and updated_cache is (K_new, V_new)
    """
    Q = x_new @ W_Q
    K = x_new @ W_K
    V = x_new @ W_V

    K = K[None,:]
    V = V[None,:]
    if cache is None:
        k_cache, v_cache = (K, V)

    else:
        k_cache, v_cache = cache
        k_cache = torch.cat([k_cache,K], dim=0)
        v_cache = torch.cat([v_cache,V], dim=0)

    score = Q @ k_cache.transpose(-1,-2) / math.sqrt(k_cache.shape[-1])
    sft_score = torch.softmax(score, dim=-1)

    return (sft_score @ v_cache, (k_cache, v_cache))







