import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here

    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(x)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    return numerator / denominator