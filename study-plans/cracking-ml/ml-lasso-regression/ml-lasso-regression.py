def lasso_regression(X, y, lr, epochs, alpha):
    """
    Perform Lasso Regression using gradient descent with L1 subgradient.
    Returns: tuple of (weights_list, bias_float)
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        y_hat = X @ w + b
        y_diff = y_hat - y

        dw = (2 / n) * (X.T @ y_diff) + alpha * np.sign(w) # np.sign()
        db = (2 / n) * np.sum(y_diff)

        w -= lr * dw
        b -= lr * db

    w = [round(float(v), 4) for v in w]
    b = round(float(b), 4)

    return w, b