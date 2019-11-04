import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def train_test_split_self(X, Y, test_ratio=0.2, seed=None):
    assert X.shape[0] == Y.shape[0], "the size of X must be equal to the size of Y."
    assert 0.0 <= test_ratio <= 1.0, "test_ratio must be valid."

    if seed:
        np.random.seed(seed)

    shuffile_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio)
    test_indexes = shuffile_indexes[:test_size]
    train_indexes = shuffile_indexes[test_size:]

    X_train = X[train_indexes]
    Y_train = Y[train_indexes]

    X_test = X[test_indexes]
    Y_test = Y[test_indexes]

    return X_train, X_test, Y_train, Y_test