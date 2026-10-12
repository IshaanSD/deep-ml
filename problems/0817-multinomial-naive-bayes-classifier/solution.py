import numpy as np

def multinomial_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Implements Multinomial Naive Bayes classifier.

    Args:
        X_train: Training count features (shape: N_train x D)
        y_train: Training labels (shape: N_train)
        X_test: Test count features (shape: N_test x D)
        alpha: Laplace smoothing parameter

    Returns:
        Predicted class labels for X_test (shape: N_test)
    """
    # Your code here
    # Fit
    priors = dict()
    logprobs = dict()
    classes = np.unique(y_train)
    for c in classes:
        X_c = X_train[y_train==c]
        priors[c] = len(X_c)/len(X_train)
        logprobs[c] = np.log(X_c.sum(axis=0) + alpha) - np.log(X_c.sum() + alpha*X_c.shape[1])
    
    # Test
    preds = []
    for x in X_test:
        scores = []
        for c in classes:
            score = priors[c]
            score += np.sum(x*logprobs[c])
            scores.append(score)
        preds.append(int(classes[np.argmax(scores)]))
    return np.array(preds)