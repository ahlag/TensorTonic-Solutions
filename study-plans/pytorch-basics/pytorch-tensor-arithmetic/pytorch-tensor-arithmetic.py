import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    x = torch.tensor(x)
    y = torch.tensor(y)

    if op == "add":
        result = torch.add(x, y)
    elif op == "multiply":
        result = torch.multiply(x, y)
    elif op == "matmul":
        result = torch.matmul(x, y)
    elif op == "power":
        result = torch.pow(x, y)
    elif op == "max":
        result = torch.max(x, y)

    return result.tolist()