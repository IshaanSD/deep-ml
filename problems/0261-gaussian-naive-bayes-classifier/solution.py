import numpy as np
import scipy.stats as ss

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
	"""
	Implements Gaussian Naive Bayes classifier.
	
	Args:
		X_train: Training features (shape: N_train x D)
		y_train: Training labels (shape: N_train)
		X_test: Test features (shape: N_test x D)
	
	Returns:
		Predicted class labels for X_test (shape: N_test)
	"""
	# Your code here
	classes = np.unique(y_train)
	params = dict()
	for c in classes:
		X_c = X_train[y_train==c]
		params[c] = len(X_c)/len(X_train), X_c.mean(axis = 0), X_c.std(axis = 0) + 1e-9

	# pred for each test example
	preds = []
	for i in range(len(X_test)):
		# score for each class
		scores = []
		for c in classes:
			prior, mean, std = params[c]
			score = np.log(prior)
			for d in range(X_test.shape[1]):
				score += ss.norm.logpdf(X_test[i,d], loc = mean[d], scale = std[d])
			scores.append(score)
		preds.append(int(classes[np.argmax(scores)]))
	return preds

