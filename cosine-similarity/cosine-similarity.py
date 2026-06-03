import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    denominator = norm_a * norm_b

    if denominator == 0:
        return 0.0

    return float(dot_product / denominator)

    