import numpy as np

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    n, d = X.shape

    weights = np.zeros(d)
    bias = 0.0

    for _ in range(n_iters):
        z = X @ weights + bias
        y_hat = 1 / (1 + np.exp(-z))
        y_diff = y_hat - y

        dw = (1.0 / n) * (X.T @ y_diff)
        db = (1.0 / n) * np.sum(y_diff)

        weights -= lr * dw
        bias -= lr * db

    weights = np.round(weights, decimals=4)
    bias = round(bias, 4)

    return weights.tolist(), bias # Use np.tolist to convert np to list
