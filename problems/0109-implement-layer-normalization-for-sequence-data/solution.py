import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	"""
	Perform Layer Normalization.
	"""
	# Your code here
	x_mean = np.mean(X, axis=-1, keepdims = True)
	x_var = np.mean((X - x_mean)**2, axis=-1, keepdims = True)
	return gamma * (X - x_mean)/np.sqrt(x_var + epsilon) + beta