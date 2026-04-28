import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)

    correct_probs = y_pred[np.arange(n), y_true]

    loss = -np.log(correct_probs)

    return np.mean(loss)