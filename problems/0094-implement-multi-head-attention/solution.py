import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    return [X @ W_q,X @ W_k,X @ W_v]
    pass

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    S = Q @ np.swapaxes(K, -2, -1) / np.sqrt(K.shape[-1])
    S = S - np.max(S, axis=-1, keepdims = True)
    sft_S = np.exp(S) / np.sum(np.exp(S), axis=-1, keepdims = True)
    return sft_S @ V
    pass

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    seq_len = Q.shape[0]
    assert Q.shape[-1] % n_heads == 0, "not divisible"
    d_k = Q.shape[-1] // n_heads
    Q = Q.reshape(Q.shape[0],n_heads,d_k).transpose(1,0,2)
    K = K.reshape(K.shape[0],n_heads,d_k).transpose(1,0,2)
    V = V.reshape(V.shape[0],n_heads,d_k).transpose(1,0,2)
    return self_attention(Q,K,V).swapaxes(0,1).reshape(seq_len,n_heads*d_k)

    # Your code here
    pass