import numpy as np

def rope(x, mode="forward", d_output=None):
    """
    Returns: Dict with "rotated", "cos_pe", "sin_pe" (and "dx" in backward mode).
    All values rounded to 4 decimal places.
    """
    x = np.asarray(x, dtype=float)

    seq_len, dim = x.shape
    half_dim = dim // 2

    positions = np.arange(seq_len)[:, None]
    inv_freq = 1.0 / (10000 ** (np.arange(half_dim) / half_dim))

    angles = positions * inv_freq

    cos_pe = np.cos(angles)
    sin_pe = np.sin(angles)

    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]

    rotated = np.zeros_like(x)
    rotated[:, 0::2] = x_even * cos_pe - x_odd * sin_pe
    rotated[:, 1::2] = x_even * sin_pe + x_odd * cos_pe

    result = {
        "rotated": np.round(rotated, 4).tolist(),
        "cos_pe": np.round(cos_pe, 4).tolist(),
        "sin_pe": np.round(sin_pe, 4).tolist(),
    }

    if mode == "backward":
        d_output = np.asarray(d_output, dtype=float)

        dy_even = d_output[:, 0::2]
        dy_odd = d_output[:, 1::2]

        dx = np.zeros_like(x)
        dx[:, 0::2] = dy_even * cos_pe + dy_odd * sin_pe
        dx[:, 1::2] = -dy_even * sin_pe + dy_odd * cos_pe

        result["dx"] = np.round(dx, 4).tolist()

    return result