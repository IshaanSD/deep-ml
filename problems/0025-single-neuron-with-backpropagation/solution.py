import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	w,b = initial_weights, initial_bias
	mse_values =[]
	for _ in range(epochs):
		x = features @ w + b # N,1
		out = 1/ (1+np.exp(-1 * x)) # sigmoid

		mse = ((out - labels)**2).mean()
		mse_values.append(round(mse,4))
		dw = np.dot(2*(out - labels)*out*(1 - out), features) / features.shape[0]
		db = np.dot(2*(out - labels)*out,(1 - out)) / features.shape[0]

		w -= learning_rate * dw
		b -= learning_rate * db

	updated_weights, updated_bias = w.round(4), round(b,4)

	return updated_weights, updated_bias, mse_values