import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    d_model = Q.shape[-1]
    head_dim = d_model // num_heads

    # Linear projections
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    # (..., seq_len, d_model) -> (..., seq_len, num_heads, head_dim)
    Q = Q.reshape(*Q.shape[:-1], num_heads, head_dim)
    K = K.reshape(*K.shape[:-1], num_heads, head_dim)
    V = V.reshape(*V.shape[:-1], num_heads, head_dim)

    # (..., seq_len, num_heads, head_dim) -> (..., num_heads, seq_len, head_dim)
    Q = np.swapaxes(Q, -3, -2)
    K = np.swapaxes(K, -3, -2)
    V = np.swapaxes(V, -3, -2)

    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(head_dim)
    weights = softmax(scores, axis=-1)
    out = np.matmul(weights, V)

    # (..., num_heads, seq_len, head_dim) -> (..., seq_len, num_heads, head_dim)
    out = np.swapaxes(out, -3, -2)

    # (..., seq_len, num_heads, head_dim) -> (..., seq_len, d_model)
    out = out.reshape(*out.shape[:-2], d_model)

    return out @ W_o