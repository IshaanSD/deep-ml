import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	features = np.array(features) # N,m
	labels = np.array(labels) # N,1
	weights = np.array(weights) # m, 1
	bias = np.array(bias) # 1,

	x = features @ weights + bias # N,1
	out = 1/ (1+np.exp(-1 * x)) # sigmoid

	mse = ((out - labels)**2).mean()

	return out.round(4), mse.round(4)

