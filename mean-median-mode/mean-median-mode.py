import numpy as np
from collections import Counter

def mean_median_mode(x):
    x = np.asarray(x)

    # mean + median
    mean = float(np.mean(x))
    median = float(np.median(x))

    # mode (smallest value if there are ties, because unique() is sorted)
    values, counts = np.unique(x, return_counts=True)
    mode = values[np.argmax(counts)]
    # convert numpy scalar to normal Python number
    mode = mode.item() if hasattr(mode, "item") else mode

    return {
        "mean": mean,
        "median": median,
        "mode": mode,
    }