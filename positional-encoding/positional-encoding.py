import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pos = np.arange(seq_len)[:, None]                 # (seq_len, 1)
    even_idx = np.arange(0, d_model, 2)               # 0,2,4,...
    denom = base ** (even_idx / d_model)              # base^(2i/d_model) where 2i = even_idx
    angles = pos / denom[None, :]                     # (seq_len, ceil(d_model/2))

    pe = np.zeros((seq_len, d_model), dtype=float)
    pe[:, even_idx] = np.sin(angles)

    odd_idx = even_idx + 1
    odd_idx = odd_idx[odd_idx < d_model]
    pe[:, odd_idx] = np.cos(angles[:, :len(odd_idx)])

    return pe