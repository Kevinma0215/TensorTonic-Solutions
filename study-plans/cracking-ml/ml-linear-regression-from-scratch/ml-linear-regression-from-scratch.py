import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    
    X = np.asarray(X)
    d = X.shape[1]
    n = X.shape[0]
    
    weights = np.zeros(d)
    bias = 0.0

    for _ in range(epochs):
        y_hat = X @ weights + bias
        y_diff = y_hat - y

        dw = (2 / n) * X.T @ (y_diff)
        db = (2 / n) * np.sum(y_diff)
        
        weights -= lr * dw
        bias -= lr * db

    return weights, bias