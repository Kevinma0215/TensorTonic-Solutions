import numpy as np

def softmax_regression(X, y, n_classes, lr=0.01, n_iters=1000):
    """
    Returns: tuple (weights, bias) where weights is a 2D list (d x K) and bias is a list of length K
    """
    
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    n, d = X.shape
    K = n_classes
    
    W = np.zeros((d, K)) # How to set 2D zeros
    b = np.zeros(K)

    # One-hot encode labels
    Y_oh = np.zeros((n, K))
    Y_oh[np.arange(n), y] = 1.0 # ??

    for _ in range(n_iters):
        Z = X @ W + b
        # Numerical stability
        Z -= Z.max(axis=1, keepdims=True) # Why?
        exp_Z = np.exp(Z)
        P = exp_Z / exp_Z.sum(axis=1, keepdims=True) # How
        
        y_diff = P - Y_oh
        dW = (1.0 / n) * (X.T @ y_diff)
        db = (1.0 / n) * y_diff.sum(axis=0)

        W -= lr * dW
        b -= lr * db

    return (W.tolist(), b.tolist())