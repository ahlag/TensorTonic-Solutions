import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    labels = np.union1d(y_true, y_pred)

    tp = 0
    fp = 0
    fn = 0
    
    for label in labels:
        tp += np.sum((y_true == label) & (y_pred == label))
        fp += np.sum((y_true != label) & (y_pred == label))
        fn += np.sum((y_true == label) & (y_pred != label))

    denominator = 2 * tp + fp + fn

    if denominator == 0:
        return 0.0

    return float((2 * tp) / denominator)