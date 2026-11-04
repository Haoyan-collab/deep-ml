import numpy as np

def transfer_and_attend(X, W_q_lin, W_k_lin, W_v_lin):
    """
    Transfer linear-style weights (shape (d_out, d_in)) into parameter-style
    weights (shape (d_in, d_out)) and compute single-head causal self-attention.

    Args:
        X: input array of shape (seq_len, d_in)
        W_q_lin, W_k_lin, W_v_lin: linear-style weight matrices of shape (d_out, d_in)

    Returns:
        Nested list of shape (seq_len, d_out): the self-attention output.
    """
    # Your code here
    X = np.array(X)
    W_k_lin = np.array(W_k_lin).T
    W_q_lin = np.array(W_q_lin).T
    W_v_lin = np.array(W_v_lin).T
    seq_len,d_in = X.shape
    Q = X@W_q_lin
    K = X@W_k_lin
    V = X@W_v_lin
    # mask
    d_out = Q.shape[-1]

    q_idx = np.arange(seq_len)[:,None]
    k_idx = np.arange(seq_len)[None,:]
    mask = np.where(k_idx>q_idx,True,False)

    z = Q @ K.swapaxes(-1,-2) / np.sqrt(d_out)
    z = np.where(mask,-1e9,z)
    z -= np.max(z,axis = -1,keepdims=True)
    z = np.exp(z)/np.sum(np.exp(z),axis=-1,keepdims = True)
    out = z@V
    return out.tolist()
