def hard_sigmoid(x: float) -> float:
	"""
	Implements the Hard Sigmoid activation function.

	Args:
		x (float): Input value

	Returns:
		float: The Hard Sigmoid of the input
	"""
	# Your code here
	import numpy as np
	return np.where(x <= -2.5,0,
					np.where(x >= 2.5, 1, 0.2*x + 0.5))
	pass