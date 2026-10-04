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
    x_mean = np.mean(x, axis=-1, keepdims= True)
    x_var = np.mean((x-x_mean)**2, axis=-1, keepdims = True)
    # layernorm
    h = (x - x_mean)/np.sqrt(x_var + 1e-5)
    h = params["ln1_gamma"] * h +params["ln1_beta"]
    # mha
    Q = h @ params["W_q"]
    K = h @ params["W_k"]
    V = h @ params["W_v"]
    
    # split head
    seq_len = Q.shape[-2]
    d_model = Q.shape[-1]
    assert d_model % num_heads == 0, "not divisible!"
    d_k = d_model // num_heads
    B = Q.shape[0]
    Q = Q.reshape(B,seq_len,num_heads,d_k).swapaxes(1,2)
    K = K.reshape(B,seq_len,num_heads,d_k).swapaxes(1,2)
    V = V.reshape(B,seq_len,num_heads,d_k).swapaxes(1,2)

    # casual mask
    q_len = Q.shape[-2]
    k_len = K.shape[-2]
    q_idx = np.arange(q_len)[:,None]
    k_idx = np.arange(k_len)[None,:]
    mask = np.where(k_idx > q_idx, -np.inf, 0)

    #attn
    score = Q @ K.swapaxes(-1,-2) / np.sqrt(d_k) + mask
    score = score - np.max(score, axis=-1,keepdims = True)
    sft_score = np.exp(score) / np.sum(np.exp(score), axis=-1, keepdims = True)
    attn_out = sft_score @ V

    # merge head + projection + residual
    attn_out = attn_out.swapaxes(1,2).reshape(B,seq_len,d_model)
    attn_out = attn_out @ params["W_o"]
    x = x + attn_out

    #layernorm 2
    x_mean = np.mean(x, axis=-1, keepdims= True)
    x_var = np.mean((x-x_mean)**2, axis=-1, keepdims = True)
    h = (x - x_mean)/np.sqrt(x_var + 1e-5)
    h = params["ln2_gamma"] * h +params["ln2_beta"] 
    #ffn 1
    h = h @ params["W_ff1"] + params["b_ff1"]
    #gelu
    h = 0.5 * h * (1 + np.tanh(np.sqrt(2/np.pi) * (h + 0.044715*h**3)))
    #ffn 2
    h = h @ params["W_ff2"] + params["b_ff2"]
    x = x + h

    return x






