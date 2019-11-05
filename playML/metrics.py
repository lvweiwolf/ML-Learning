import numpy as np

def accuracy_score(Y_true, Y_predict):
    assert Y_true.shape[0] == Y_predict.shape[0], "the size of Y_true must equal to the size of Y_predict."
    return sum(Y_true == Y_predict) / len(Y_true)