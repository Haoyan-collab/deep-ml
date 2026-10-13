import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def attention(q, k, v, key_padding_mask=None, causal=False):
    # TODO: build a single bool attn_mask broadcastable to (B, H, T_q, T_k)
    #       where True means "may attend"
    # TODO: combine the causal lower-triangular mask and the key padding mask
    # TODO: return F.scaled_dot_product_attention(q, k, v, attn_mask=...)
    # score = q @ k.transpose(-1,-2) / math.sqrt(k.shape[-1])
    batch = q.shape[0]
    seq = q.shape[-2]

    # calculate mask, true means valid
    c_mask = None
    if causal:
        q_len = q.shape[-2]
        k_len = k.shape[-2]
        q_idx = torch.arange(q_len)[:,None]
        k_idx = torch.arange(k_len)[None,:]
        c_mask = torch.where(k_idx > q_idx, False, True) #(t_q, t_k)

    if key_padding_mask is not None:  #(batch, t_k)
        key_padding_mask = key_padding_mask.reshape(batch,1,1,-1)

    attn_mask = None
    if c_mask is not None and key_padding_mask is not None:
        attn_mask = key_padding_mask & c_mask
    
    elif c_mask is not None:
        attn_mask = c_mask

    elif key_padding_mask is not None:
        attn_mask = key_padding_mask

    # mask score
    # score = torch.where(attn_mask, score, -torch.inf)
    
    # score = torch.softmax(score, dim=-1)
    return F.scaled_dot_product_attention(q,k,v,attn_mask = attn_mask)






