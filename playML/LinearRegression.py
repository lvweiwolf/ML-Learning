import numpy as np
from .metrics import r2_score

class LinearRegression:
    
    def __init__(self):
        self.coef_ = None
        self.interception_ = None   # 截距 θ0
        self._theta = None          # 系数向量

    def fit_normal(self, X_train, Y_train):
        assert X_train.shape[0] == Y_train.shape[0], "the size of X_train must be equal to the size of Y_train."

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y_train)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        assert self.interception_ is not None and self.coef_ is not None,\
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_),\
            "the feature number of X_predict must be equal to X_train"
        
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, Y_test):
        Y_predict = self.predict(X_test)
        return r2_score(Y_test, Y_predict)

    def __repr__(self):
        return "LinearRegression()"
    