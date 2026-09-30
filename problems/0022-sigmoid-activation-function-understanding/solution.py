import math
import numpy as np
def sigmoid(z: float) -> float:
	#Your code here
	if z >= 0:
		result = 1 / (1 + np.exp(-z))
	else:
		result = np.exp(z) / (1 + np.exp(z))
	return result