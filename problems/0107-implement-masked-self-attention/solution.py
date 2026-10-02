import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
	# Your code here
	S = Q @ np.swapaxes(K, -1, -2) / np.sqrt(K.shape[-1])
	S += mask

	max_s = np.max(S, axis=-1, keepdims = True)
	# max_s = np.where(max_s == -np.inf, 0, max_s)
	S -= max_s
	sft_S = np.exp(S) / np.sum(np.exp(S), axis=-1, keepdims = True)
	return sft_S @ V
	pass