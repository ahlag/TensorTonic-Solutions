import torch
import torch.nn.functional as F

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    x = torch.tensor(x, dtype=torch.float32)
    method = method.lower()

    if method == "relu":
        out = torch.relu(x)
    elif method == "sigmoid":
        out = torch.sigmoid(x)
    elif method == "tanh":
        out = torch.tanh(x)
    elif method in ["leakyrelu", "leaky_relu"]:
        out = F.leaky_relu(x, negative_slope=0.01)
    else:
        raise ValueError(f"Unsupported activation method: {method}")

    return out.tolist()