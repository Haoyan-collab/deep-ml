import torch
import torch.nn as nn

def count_params():
    """Build Sequential(Linear(4,8), ReLU, Linear(8,2)) and return trainable param count.

    Returns:
        int: total number of trainable parameters
    """
    # TODO: build the model and sum p.numel() for trainable params
    model = nn.Sequential(
        nn.Linear(4,8),
        nn.ReLU(),
        nn.Linear(8,2)
    )

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
