import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Convert to numpy array (in case it's a list)
    y_train = np.array(y_train)

    # Find unique labels and their counts
    values, counts = np.unique(y_train, return_counts=True)

    # Get the label with maximum count
    majority_label = values[np.argmax(counts)]

    print(majority_label)

    return np.array([majority_label] * len(X_test))