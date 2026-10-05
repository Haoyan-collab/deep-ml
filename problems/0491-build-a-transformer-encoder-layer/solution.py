import numpy as np

def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        weights: Dictionary containing all weight matrices and normalization parameters
        num_heads: Number of attention heads
        eps: Epsilon for layer normalization

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    # Your code here
    #layernorm 1
    # x_mean = np.mean(X, axis=-1,keepdims = True)
    # x_var = np.mean((X - x_mean)**2, axis=-1, keepdims = True)
    # h = weights['gamma1'] * (X-x_mean) / np.sqrt(x_var + eps) + weights['beta1']
    h = X

    Q = h @ weights['W_q']
    K = h @ weights['W_k']
    V = h @ weights['W_v']

    #split num_heads
    seq_len = Q.shape[-2]
    d_model = Q.shape[-1]
    assert d_model % num_heads == 0, "not divisible"
    d_k = d_model // num_heads

    Q = Q.reshape(Q.shape[0],seq_len, num_heads, d_k).swapaxes(1,2)
    K = K.reshape(K.shape[0],seq_len, num_heads, d_k).swapaxes(1,2)
    V = V.reshape(V.shape[0],seq_len, num_heads, d_k).swapaxes(1,2)

    #attn
    scores = Q @ K.swapaxes(-1,-2) / np.sqrt(d_k)
    scores -= np.max(scores, axis=-1, keepdims = True)
    attn = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims = True)
    h = attn @ V

    #merge num_heads
    h = h.swapaxes(1,2).reshape(Q.shape[0], seq_len, d_model)
    z = X + h @ weights['W_o']

    #layernorm 2
    z_mean = np.mean(z, axis=-1,keepdims = True)
    z_var = np.mean((z - z_mean)**2, axis=-1, keepdims = True)
    z = weights['gamma1'] * (z-z_mean) / np.sqrt(z_var + eps) + weights['beta1']

    #FFN
    H = z @ weights['W1'] + weights['b1']
    #relu
    H = np.where(H > 0, H, 0)

    H = H @ weights['W2'] + weights['b2']

    z = z + H

    z_mean = np.mean(z, axis=-1,keepdims = True)
    z_var = np.mean((z - z_mean)**2, axis=-1, keepdims = True)
    z = weights['gamma2'] * (z-z_mean) / np.sqrt(z_var + eps) + weights['beta2']
    return z










    pass