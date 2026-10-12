import torch
import math
def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                           block_size: int = 2) -> torch.Tensor:
    """
    Compute attention output using Flash Attention v1 algorithm.
    
    Args:
        Q: Query matrix (seq_len, d_model)
        K: Key matrix (seq_len, d_model)
        V: Value matrix (seq_len, d_model)
        block_size: Size of blocks for tiled computation
    
    Returns:
        Output matrix (seq_len, d_model)
    """
    seq_len, d_model = Q.shape
    out = torch.zeros((seq_len,d_model), dtype = Q.dtype, device = Q.device)

    # for each query block_size
    for i in range(0,seq_len,block_size):
        q = Q[i:i+block_size,:]
        bq = q.shape[0]
        
        # set up online softmax params
        m = torch.full((bq,1), -float("inf"))
        l = torch.zeros((bq,1))
        acc = torch.zeros((bq,d_model))

        # check each k,v
        for j in range(0,seq_len,block_size):
            k = K[j:j+block_size,:]
            v = V[j:j+block_size,:]

            scores = q @ k.transpose(-1,-2) / math.sqrt(d_model)

            # update params
            new_m = torch.maximum(m,scores.max(dim = -1, keepdim = True).values)
            alpha = torch.exp(m-new_m)

            scores = torch.exp(scores - new_m)
            l = alpha * l + torch.sum(scores, dim=-1, keepdim = True)
            acc = alpha * acc + scores @ v

            m = new_m

        out[i:i+bq,:] = acc/l

    return out



















