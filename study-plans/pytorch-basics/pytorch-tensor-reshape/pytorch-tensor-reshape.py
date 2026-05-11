import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    x = torch.tensor(x, dtype=torch.float32)
    if op == 'flatten':
        return torch.flatten(x)
    elif op == 'squeeze':
        return torch.squeeze(x)
    elif op == 'transpose':
        return torch.transpose(x, 0, 1)
