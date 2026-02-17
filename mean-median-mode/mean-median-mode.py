import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here

    values, counts = np.unique(np.array(x), return_counts=True)
    mode = values[np.argmax(counts)]
    
    return {
        'mean': np.mean(x),
        'median': np.median(x),
        'mode': mode
    }