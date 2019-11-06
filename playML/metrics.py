import numpy as np

def accuracy_score(Y_true, Y_predict):
    assert Y_true.shape[0] == Y_predict.shape[0], "the size of Y_true must equal to the size of Y_predict."
    return sum(Y_true == Y_predict) / len(Y_true)

def mean_squared_error(y_true, Y_predict):
    assert len(y_true) == len(Y_predict), "the size of y_true must be equal to the size of y_predict."

    return np.sum((y_true - Y_predict)**2) / len(y_true)

def root_mean_squared_error(y_true, y_predict):
    return np.sqrt(mean_squared_error(y_true, y_predict))

def mean_absolute_error(y_true, y_predict):
    assert len(y_true) == len(y_predict), "the size of y_true must be equal to the size of y_predict."

    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)

def r2_score(y_true, y_predict):
    assert len(y_true) == len(y_predict), "the size of y_true must be equal to the size of y_predict."

    return 1.0 - mean_squared_error(y_true, y_predict) / np.var(y_true)