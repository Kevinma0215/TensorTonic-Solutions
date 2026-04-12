import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """

    # Use np.array to copy list to np array
    X = np.array(X, dtype=float) 
    y = np.array(y, dtype=float)

    # shape can be use as item
    n, d = X.shape
    
    weights = np.zeros(d)
    bias = 0.0

    for _ in range(epochs):
        # Notice inner product use @
        y_hat = X @ weights + bias
        y_diff = y_hat - y

        dw = (2 / n) * (X.T @ y_diff)
        db = (2 / n) * np.sum(y_diff)
        
        weights -= lr * dw
        bias -= lr * db

    # Remember to round every time before return
    weights = np.round(weights, decimals=4)
    bias = round(bias, 4)

    return weights, bias