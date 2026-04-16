import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None, mode="forward", d_output=None):
    """
    Returns: Dict with "output", "attention_weights", and optionally "dQ", "dK", "dV".
    """
    Q = np.array(Q, dtype=float)
    K = np.array(K, dtype=float)
    V = np.array(V, dtype=float)

    d_k = Q.shape[-1] # the last index
    scale = np.sqrt(d_k)

    scores = Q @ K.T / scale

    if mask is not None:
        mask = np.array(mask, dtype=bool)
        scores = np.where(mask, scores, -1e9)

    e = np.exp(scores - np.max(scores, axis=-1, keepdims=True)) # axis=-1 -> the last dim -> feature
    weights = e / np.sum(e, axis=-1, keepdims=True)

    output = weights @ V

    result = {
        "output": [[round(float(v), 4) for v in row] for row in output],
        "attention_weights": [[round(float(v), 4) for v in row] for row in weights]
    }

    if mode == "backward" and d_output is not None:
        d_output = np.array(d_output, dtype=float)
        dV = weights.T @ d_output
        d_weights = d_output @ V.T
        d_scores = weights * (d_weights - np.sum(d_weights * weights, axis=-1, keepdims=True))
        if mask is not None:
            d_scores = np.where(mask, d_scores, 0.0)
        dQ = d_scores @ K / scale
        dK = d_scores.T @ Q / scale
        result["dQ"] = [[round(float(v), 4) for v in row] for row in dQ]
        result["dK"] = [[round(float(v), 4) for v in row] for row in dK]
        result["dV"] = [[round(float(v), 4) for v in row] for row in dV]

    return result