import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    # n_rows, n_cols = len(A), len(A[0])
    # [list(row) for row in zip(*A)]
    A = np.asarray(A)
    return A.T
    
