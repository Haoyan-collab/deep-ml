import math
import numpy as np
def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	y = np.array(features) @ np.array(weights) + np.array(bias)
	probabilities = np.where(
		y >= 0,
		1 / (1 + np.exp(-y)),
		np.exp(y) / (1 + np.exp(y))
	)
	labels = np.array(labels)
	mse = np.mean((probabilities - labels) ** 2)
	mse = np.round(float(mse), 4)
	probabilities = np.round((probabilities), 4).tolist()
	return probabilities, mse