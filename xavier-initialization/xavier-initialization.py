def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    # Write code here
    L = math.sqrt(6 / (fan_in + fan_out))

    return [
        [w * 2 * L - L for w in row] for row in W
    ]