import numpy as np
import pandas as pd

def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True):
    """
    Split data into training and test sets.

    Parameters:
    - X: features (DataFrame or numpy array)
    - y: target (Series or numpy array)
    - test_size: proportion of test set (e.g. 0.2 for 20%)
    - random_state: for reproducibility
    - shuffle: whether to shuffle the data before splitting

    Returns:
    - X_train, X_test, y_train, y_test
    """
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X = X.values
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.values

    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.RandomState(seed=random_state)
        rng.shuffle(indices)

    test_size = int(n_samples * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test
