import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	# Your code here
	h = x @ w1
	h = np.where(h>0, h, 0)
	h = h @ w2
	result = h + x
	return np.where(result > 0, result, 0)