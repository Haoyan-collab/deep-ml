import torch

def rnn_forward(input_sequence: list, initial_hidden_state: list, Wx: list, Wh: list, b: list) -> torch.Tensor:
    """
    Implements a simple RNN cell forward pass using PyTorch.

    Args:
        input_sequence: List of input vectors for each time step.
        initial_hidden_state: The initial hidden state vector.
        Wx: Weight matrix for input-to-hidden connections.
        Wh: Weight matrix for hidden-to-hidden connections.
        b: Bias vector.

    Returns:
        torch.Tensor: The final hidden state after processing the entire sequence,
                      rounded to four decimal places.
    """
    n = len(input_sequence)
    h = torch.tensor(initial_hidden_state)
    Wx = torch.tensor(Wx)
    Wh = torch.tensor(Wh)
    b = torch.tensor(b)
    for i in range(n):
        x = torch.tensor(input_sequence[i])
        z = Wx @ x + Wh @ h + b
        h = torch.tanh(z)
    return h