import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # TODO: store d_head, create q/k/v/out projections (all bias=False)
        assert d_model % num_heads == 0, "not divisible!"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.qproj = nn.Linear(d_model,d_model,bias = False)
        self.kproj = nn.Linear(d_model,d_model,bias = False)
        self.vproj = nn.Linear(d_model,d_model,bias = False)
        self.oproj = nn.Linear(d_model,d_model,bias = False)

    def forward(self, x, mask=None):
        # x: (B, T, d_model); mask: (T, T) of 0 and -inf, or None
        # TODO: project, reshape into heads, scaled dot-product, mask, softmax, combine, reshape back, out_proj
        q = self.qproj(x)
        k = self.kproj(x)
        v = self.vproj(x)

        # split heads
        t = q.shape[-2]
        q = q.reshape(q.shape[0],t,self.num_heads,self.d_head).transpose(1,2)
        k = k.reshape(q.shape[0],t,self.num_heads,self.d_head).transpose(1,2)
        v = v.reshape(q.shape[0],t,self.num_heads,self.d_head).transpose(1,2)

        score = q @ k.transpose(-1,-2) / math.sqrt(self.d_head)
        if mask is not None:
            score += mask
        sft_score = torch.softmax(score, dim=-1)
        out = sft_score @ v

        # merge
        out = out.transpose(1,2).reshape(q.shape[0],t,self.d_model)
        return self.oproj(out)









