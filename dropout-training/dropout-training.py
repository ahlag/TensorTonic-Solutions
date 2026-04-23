import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x, dtype=float)

    if not (0.0 <= p < 1.0):
        raise ValueError("p must be in [0, 1).")

    if rng is None:
        rng = np.random.default_rng()

    keep_prob = 1.0 - p
    mask = (rng.random(x.shape) >= p).astype(float)
    dropout_pattern = mask / keep_prob
    output = x * dropout_pattern

    return output, dropout_pattern