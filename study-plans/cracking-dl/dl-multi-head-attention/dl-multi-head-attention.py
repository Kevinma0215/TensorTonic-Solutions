import numpy as np

def multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads, mask=None):
    """
    Returns: Dict with "output" and "attention_weights", rounded to 4 decimal places.
    """
    Q = np.array(Q, dtype=float)
    K = np.array(K, dtype=float)
    V = np.array(V, dtype=float)
    W_q = np.array(W_q, dtype=float)
    W_k = np.array(W_k, dtype=float)
    W_v = np.array(W_v, dtype=float)
    W_o = np.array(W_o, dtype=float)

    seq_len, d_model = Q.shape
    d_head = d_model // num_heads

    # Project Q, K, V
    Q_proj = Q @ W_q.T
    K_proj = K @ W_k.T
    V_proj = V @ W_v.T

    # Split into heads
    Q_heads = Q_proj.reshape(seq_len, num_heads, d_head).transpose(1, 0, 2)
    K_heads = K_proj.reshape(seq_len, num_heads, d_head).transpose(1, 0, 2)
    V_heads = V_proj.reshape(seq_len, num_heads, d_head).transpose(1, 0, 2)

    scale = np.sqrt(d_head)
    all_head_outputs = []
    all_attention_weights = []

    for h in range(num_heads):
        Q_h = Q_heads[h]
        K_h = K_heads[h]
        V_h = V_heads[h]

        scores = Q_h @ K_h.T / scale

        if mask is not None:
            mask_arr = np.array(mask, dtype=bool)
            scores = np.where(mask_arr, scores, -1e9)

        # stable softmax
        e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = e / np.sum(e, axis=-1, keepdims=True)

        head_out = weights @ V_h
        all_head_outputs.append(head_out)
        all_attention_weights.append(weights)

    # Concatenate heads and project
    concat = np.concatenate(all_head_outputs, axis=-1)
    output = concat @ W_o.T

    return {
        "output": [[round(float(v), 4) for v in row] for row in output],
        "attention_weights": [[[round(float(v), 4) for v in row] for row in hw] for hw in all_attention_weights]
    }