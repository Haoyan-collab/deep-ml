import numpy as np

def LN(z: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: np.ndarray) -> np.ndarray:
	z_mean = np.mean(z, axis=-1, keepdims = True)
	z_var = np.mean((z - z_mean)**2, axis=-1, keepdims = True)
	return gamma * (z-z_mean)/np.sqrt(z_var + eps) + beta

def transformer_block(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray, mode: str, eps: float = 1e-5) -> np.ndarray:
	"""
	Apply a transformer block with two sublayers using either pre-norm or post-norm.
	
	Args:
		x: Input array of shape (seq_len, d_model)
		W1, b1: Weights and bias for first sublayer
		W2, b2: Weights and bias for second sublayer
		gamma1, beta1: LayerNorm params for first normalization
		gamma2, beta2: LayerNorm params for second normalization
		mode: 'pre_norm' or 'post_norm'
		eps: Epsilon for numerical stability
	
	Returns:
		Output array of shape (seq_len, d_model)
	"""
	if mode == 'post_norm':
		h = x + x @ W1 + b1
		h = LN(h,gamma1,beta1,eps)
		out = h + h @ W2 + b2
		return LN(out,gamma2,beta2,eps)
	elif mode == 'pre_norm':
		h = LN(x,gamma1,beta1,eps)
		h = x + h @ W1 + b1
		out = LN(h,gamma2,beta2,eps)
		out = h + out @ W2 + b2
		return out
	pass