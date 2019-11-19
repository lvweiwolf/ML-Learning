import numpy as np
from .metrics import r2_score

class SimpleLinearRegression1:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, X_train, Y_train):
        assert X_train.ndim == 1, "Simple Linear Regressor can only solve single feature training data."
        assert len(X_train) == len(Y_train), "the size of X_train must be equal to the size of Y_train."

        x_mean = np.mean(X_train)
        y_mean = np.mean(Y_train)

        num = 0.0
        d = 0.0
        num = np.sum((X_train - x_mean)*(Y_train - y_mean))
        d = np.sum((X_train - x_mean) ** 2)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        
        return self

    def predict(self, X_predict):
        assert X_predict.ndim == 1, "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, "must fit before predict!"

        return np.array([self._predict(x) for x in X_predict])

    def _predict(self, X_single):
        return self.a_ * X_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1()"


class SimpleLinearRegression2:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, X_train, Y_train):
        assert X_train.ndim == 1, "Simple Linear Regressor can only solve single feature training data."
        assert len(X_train) == len(Y_train), "the size of X_train must be equal to the size of Y_train."

        x_mean = np.mean(X_train)
        y_mean = np.mean(Y_train)

        num = 0.0
        d = 0.0
        num = (X_train - x_mean).dot(Y_train - y_mean)
        d = (X_train - x_mean).dot(X_train - x_mean)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        
        return self
        

    def predict(self, X_predict):
        assert X_predict.ndim == 1, "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, "must fit before predict!"

        return np.array([self._predict(x) for x in X_predict])

    def score(self, X_test, Y_test):
        Y_predict = self.predict(X_test)
        return r2_score(Y_test, Y_predict)

    def _predict(self, X_single):
        return self.a_ * X_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression2()"