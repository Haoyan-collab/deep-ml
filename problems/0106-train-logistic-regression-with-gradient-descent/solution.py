import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	# Your code here
	n,d = X.shape
	X = np.column_stack([np.ones(n),X])
	theta = np.zeros(d+1)
	loss_lst = []
	for _ in range(iterations):
		z = X@ theta
		sig_z = np.where(z>0, 
					1/(1+np.exp(-z)), 
					np.exp(z)/(1+np.exp(z)))
		loss_lst.append(-np.sum(y*np.log(sig_z)+(1-y)*np.log(1-sig_z), axis=-1))
		theta = theta - learning_rate*(X.T@(sig_z-y))
	return (np.round(theta, 4).tolist(),np.round(loss_lst, 4).tolist())