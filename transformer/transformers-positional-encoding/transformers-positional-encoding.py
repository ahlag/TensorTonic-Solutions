import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    pe = np.zeros((seq_length, d_model))
    position = np.arange(seq_length).reshape(-1, 1)
    even_idx = np.arange(0, d_model, 2)
    div_term = np.exp(even_idx * (-np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term[:pe[:, 1::2].shape[1]])

    return pe
    