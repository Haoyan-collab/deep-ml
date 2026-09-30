import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# Your code here
	max_score = max(scores)
	lg_sm = np.log(sum(np.exp(x - max_score) for x in scores))
	return [x - max_score - lg_sm for x in scores]
	pass