import torch
def pos_encoding(position: int, d_model: int):
    """
    Compute positional encodings for Transformer models.

    Args:
        position: sequence length (number of positions)
        d_model: model dimensionality

    Returns:
        torch.Tensor of shape (position, d_model) with dtype float16,
        or -1 if position == 0 or d_model <= 0.
    """
    # Your code here
    if position == 0 or d_model <= 0:
        return -1

    # position -> (position, 1)
    pos = torch.arange(position)[:,None]

    # dimension coding
    dims = torch.arange(d_model)[0::2]  #(d_model//2,)

    angles = pos / (10000**(dims/d_model))
    pe = torch.zeros((position, d_model))

    pe[:,0::2] = torch.sin(angles)
    pe[:,1::2] = torch.cos(angles[:,0:pe[:,1::2].shape[-1]])

    return pe.to(torch.float16)