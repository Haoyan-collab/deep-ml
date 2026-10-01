import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
	"""
	Compute Scaled Dot-Product Attention.
	
	Args:
		Q: Query matrix of shape (seq_len_q, d_k)
		K: Key matrix of shape (seq_len_k, d_k)
		V: Value matrix of shape (seq_len_k, d_v)
		mask: Optional binary mask of shape (seq_len_q, seq_len_k)
	
	Returns:
		Tuple of (output, attention_weights)
	"""
	# Your code here
	S = (Q @ K.T) / np.sqrt(K.shape[-1])
	if mask is not None:
		S = np.where(mask == 1, S, -np.inf)
	S -= np.max(S, axis=-1, keepdims = True)
	Weights = np.exp(S) / np.sum(np.exp(S), axis=1, keepdims = True)
	output = Weights @ V
	return (output, Weights)
	pass