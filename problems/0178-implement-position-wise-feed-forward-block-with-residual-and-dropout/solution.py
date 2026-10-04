import numpy as np

def ffn(x: list[float], W1: list[list[float]], b1: list[float], W2: list[list[float]], b2: list[float], dropout_p: float=0.1, seed: int=42) -> list[float]:
	"""
	Implement a position-wise feed-forward block with residual and dropout.

	Args:
		x: input vector
		W1, b1: first linear layer parameters
		W2, b2: second linear layer parameters
		dropout_p: dropout probability
		seed: random seed for reproducibility

	Returns:
		Output vector after FFN block (rounded to 4 decimals)
	"""
	# Your code here
	x = np.array(x)
	W1 = np.array(W1)
	b1 = np.array(b1)
	W2 = np.array(W2)
	b2 = np.array(b2)

	hidden = x @ W1.T + b1
	hidden = np.where(hidden > 0, hidden, 0)

	output = hidden @ W2.T + b2

	np.random.seed(seed)
	mask = np.random.rand(*output.shape) >= dropout_p
	output = output * mask / (1 - dropout_p)
	return output + x

	pass