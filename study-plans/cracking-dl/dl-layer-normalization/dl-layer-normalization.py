import numpy as np

def layer_normalization(x, gamma, beta, eps=1e-5, mode="forward", d_output=None):
    """
    Returns: Dict with "output", "mean", "var", "x_hat",
    and optionally "dx", "dgamma", "dbeta".
    """

    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)

    # For returning
    mean = np.mean(x, axis=-1)
    var = np.var(x, axis=-1)

    # For broadcasting calculation
    mean_keep = np.mean(x, axis=-1, keepdims=True)
    var_keep = np.var(x, axis=-1, keepdims=True)

    x_hat = (x - mean_keep) / np.sqrt(var_keep + eps)
    output = gamma * x_hat + beta

    result = {
        "output": np.round(output, 4),
        "mean": np.round(mean, 4),
        "var": np.round(var, 4),
        "x_hat": np.round(x_hat, 4)
    }

    if mode == "backward":
        if d_output is None:
            raise ValueError("d_output is required for backward mode")

        d_output = np.asarray(d_output, dtype=float)

        D = x.shape[-1]

        dbeta = np.sum(d_output, axis=0)
        dgamma = np.sum(d_output * x_hat, axis=0)

        dx_hat = d_output * gamma
        inv_std = 1.0 / np.sqrt(var_keep + eps)

        dx = (1.0 / D) * inv_std * (
            D * dx_hat
            - np.sum(dx_hat, axis=-1, keepdims=True)
            - x_hat * np.sum(dx_hat * x_hat, axis=-1, keepdims=True)
        )

        result["dx"] = np.round(dx, 4)
        result["dgamma"] = np.round(dgamma, 4)
        result["dbeta"] = np.round(dbeta, 4)

    return result