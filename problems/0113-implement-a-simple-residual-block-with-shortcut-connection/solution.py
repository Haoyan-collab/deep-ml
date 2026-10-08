import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	# Your code here
	h = w1 @ x
	h = np.where(h>0, h, 0)
	h = w2 @ h
	result = h + x
	return np.where(result > 0, result, 0)