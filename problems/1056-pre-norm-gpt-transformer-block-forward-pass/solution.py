import numpy as np

def pre_norm_transformer_block(x, params, num_heads):
    """
    Pre-norm Transformer block forward pass.

    Args:
        x: numpy array of shape (batch, seq_len, emb_dim)
        params: dict with keys 'ln1_gamma','ln1_beta','ln2_gamma','ln2_beta',
                'W_q','W_k','W_v','W_o','W_ff1','b_ff1','W_ff2','b_ff2'
        num_heads: int, number of attention heads (emb_dim must be divisible by num_heads)

    Returns:
        numpy array of shape (batch, seq_len, emb_dim)
    """
    #pre_layernorm
    x_mean = np.mean(x, axis=-1,keepdims = True)
    x_var = np.mean((x - x_mean)**2, axis=-1, keepdims = True)

    h = params['ln1_gamma'] * (x - x_mean) / np.sqrt(x_var + 1e-5) + params['ln1_beta']

    Q = h @ params['W_q']
    K = h @ params['W_k']
    V = h @ params['W_v']
    #split num_heads
    seq_len = Q.shape[-2]
    d_model = Q.shape[-1]
    assert d_model % num_heads == 0, "nod divisible!"
    d_k = d_model // num_heads
    Q = Q.reshape(Q.shape[0],seq_len,num_heads,d_k).swapaxes(1,2)
    K = K.reshape(K.shape[0],seq_len,num_heads,d_k).swapaxes(1,2)
    V = V.reshape(V.shape[0],seq_len,num_heads,d_k).swapaxes(1,2)

    #construct mask
    q_len = Q.shape[-2]
    k_len = K.shape[-2]
    q_idx = np.arange(q_len)[:,None]
    k_idx = np.arange(k_len)[None,:]

    mask = np.where(k_idx > q_idx, -np.inf, 0)

    #attn
    scores = Q @ K.swapaxes(-1,-2) / np.sqrt(d_k) + mask
    scores -= np.max(scores, axis=-1,keepdims = True)
    h = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims = True)
    h = h @ V

    #merge num_heads
    h = h.swapaxes(1,2).reshape(h.shape[0],seq_len,d_model)
    #projection
    h = h @ params['W_o']
    x = h + x

    #pre_layernorm 2
    x_mean = np.mean(x, axis=-1,keepdims = True)
    x_var = np.mean((x - x_mean)**2, axis=-1, keepdims = True)

    z = params['ln2_gamma'] * (x - x_mean) / np.sqrt(x_var + 1e-5) + params['ln2_beta']

    # ffn 1
    z = z @ params['W_ff1'] + params['b_ff1']
    #gelu
    z = 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi)) * (z + 0.044715*z**3))
    # ffn 2
    z = z @ params['W_ff2'] + params['b_ff2']

    return x + z








