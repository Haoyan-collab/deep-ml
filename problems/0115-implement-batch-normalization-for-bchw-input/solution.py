import torch
import torch.nn.functional as F
def batch_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """Perform Batch Normalization on a 4D tensor in BCHW format."""
    return F.batch_norm(
        X,
        running_mean=None,
        running_var=None,
        weight=gamma,
        bias=beta,
        training=True,
        eps=epsilon
    )