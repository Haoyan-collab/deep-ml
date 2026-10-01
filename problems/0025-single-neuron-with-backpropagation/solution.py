import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	mse_values = []
	n = len(labels)
	for _ in range(epochs):
		z = features @ initial_weights + initial_bias
		pred = np.where(
			z >= 0,
			1 / (1 + np.exp(-z)),
			np.exp(z) / (1 + np.exp(z))
		)
		mse = np.mean((pred - labels)**2)
		mse_values.append(np.round(mse, 4))
		
		dz = 2 / n * (pred - labels) * pred * (1 - pred)
		dw = np.swapaxes(features, -2, -1) @ dz
		db = sum(dz)

		initial_weights -= learning_rate * dw
		initial_bias -= learning_rate * db
	updated_weights = np.round(initial_weights, 4).tolist()
	updated_bias = np.round(initial_bias, 4)

	return updated_weights, updated_bias, mse_values