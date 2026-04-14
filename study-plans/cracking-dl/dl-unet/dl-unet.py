import numpy as np

def unet_skip(x, W_down, b_down, W_up, b_up, W_out, b_out):
    """
    Returns: Dict with "encoded", "decoded", "combined", "output", values rounded to 4 decimals.
    """
    re = {}
    
    x = np.array(x, dtype=float)
    
    enc = np.maximum(W_down @ x + b_down, 0)
    re["encoded"] = enc
    
    dec = np.maximum(W_up @ enc + b_up, 0)
    re["decoded"] = dec
    
    combined = dec + x
    re["combined"] = combined

    out = W_out @ combined + b_out
    re["output"] = out

    return re