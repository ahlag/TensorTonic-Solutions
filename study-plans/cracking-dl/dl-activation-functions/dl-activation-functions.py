import numpy as np

def activation_functions(x, activation):
    """
    Returns: [activation_value, derivative_value]
    """
    x = float(x)
    activation = activation.lower()

    if activation == "relu":
        y = max(0.0, x)
        dy = 1.0 if x > 0 else 0.0

    elif activation == "leaky_relu":
        y = x if x > 0 else 0.01 * x
        dy = 1.0 if x > 0 else 0.01

    elif activation == "sigmoid":
        y = 1 / (1 + np.exp(-x))
        dy = y * (1 - y)

    elif activation == "tanh":
        y = np.tanh(x)
        dy = 1 - y ** 2

    elif activation == "gelu":
        k = np.sqrt(2 / np.pi)
        u = k * (x + 0.044715 * x ** 3)

        y = 0.5 * x * (1 + np.tanh(u))

        dy = (
            0.5 * (1 + np.tanh(u))
            + 0.5 * x * (1 - np.tanh(u) ** 2)
            * k * (1 + 3 * 0.044715 * x ** 2)
        )

    elif activation == "swish":
        sigmoid = 1 / (1 + np.exp(-x))
        y = x * sigmoid
        dy = sigmoid + x * sigmoid * (1 - sigmoid)

    else:
        raise ValueError("Unknown activation function")

    return [round(y, 4), round(dy, 4)]