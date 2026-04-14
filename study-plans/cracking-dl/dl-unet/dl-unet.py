import numpy as np

def unet_skip(x, W_down, b_down, W_up, b_up, W_out, b_out):
    """
    Returns: Dict with "encoded", "decoded", "combined", "output", values rounded to 4 decimals.
    """
    re = {}
    
    x = np.array(x, dtype=float)
    W_down = np.array(W_down, dtype=float)
    b_down = np.array(b_down, dtype=float)
    W_up = np.array(W_up, dtype=float)
    b_up = np.array(b_up, dtype=float)
    W_out = np.array(W_out, dtype=float)
    b_out = np.array(b_out, dtype=float)
    
    enc = np.maximum(W_down @ x + b_down, 0)
    re["encoded"] = [round(float(v), 4) for v in enc]
    
    dec = np.maximum(W_up @ enc + b_up, 0)
    re["decoded"] = [round(float(v), 4) for v in dec]
    
    combined = dec + x
    re["combined"] = [round(float(v), 4) for v in combined]

    out = W_out @ combined + b_out
    re["output"] = [round(float(v), 4) for v in out]

    return re