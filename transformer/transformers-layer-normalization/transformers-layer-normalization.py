import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    # Your code here
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    x_norm = gamma * (x - mean) / np.sqrt(var + eps) + beta

    return x_norm