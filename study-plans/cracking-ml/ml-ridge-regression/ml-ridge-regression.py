def ridge_regression(X, y, lr, epochs, alpha):
    """
    Perform ridge regression using gradient descent.
    Returns: tuple of (weights_list, bias)
    """
    # Preprosecc input
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # init w, b
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    
    # training loop
    for _ in range(epochs):
        # Cal y_hat
        y_hat = X @ w + b
        # Cal y_diff
        y_diff = y_hat - y

        # Cal dw
        dw = (2.0 / n) * (X.T @ y_diff) + 2.0 * alpha * w
        # Cal db
        db = (2.0 / n) * np.sum(y_diff)

        # update w
        w -= lr * dw
        # update b
        b -= lr * db

    w = [round(float(v), 4) for v in w]
    b = round(float(b), 4)
    
    return w, b