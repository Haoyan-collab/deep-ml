import numpy as np

def attention_dropout(attn_weights, values, dropout_rate, mask):
    """Apply dropout mask to attention weights and compute context vectors."""
    attn_weights = np.array(attn_weights)
    values = np.array(values)
    mask = np.array(mask)
    attn_weights = np.where(mask == 0, 
                0, attn_weights / (1 - dropout_rate))
    return attn_weights @ values
    pass
