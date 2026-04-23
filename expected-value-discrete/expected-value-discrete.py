import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here

    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    if x.shape != p.shape:
        raise ValueError("x and p must have the same shape.")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("Probabilities must be between 0 and 1.")
    if not np.isclose(np.sum(p), 1.0):
        raise ValueError("Probabilities must sum to 1.")
    return np.sum(x * p)
