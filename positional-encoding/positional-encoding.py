import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    pe = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, np.newaxis]

    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(base) / d_model))

    pe[:, 0::2] = np.sin(pos * div_term)
    pe[:, 1::2] = np.cos(pos * div_term[:pe[:, 1::2].shape[1]])

    return pe