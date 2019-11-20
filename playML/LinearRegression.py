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

    
    def fit_gd(self, X_train, Y_train, eta=0.01, n_iters=1e4):
        assert X_train.shape[0] == Y_train.shape[0], "the size of X_train must be equal to the size of Y_train. "

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')
        
        def dJ(theta, X_b, y):
            # res = np.empty(len(theta))
            # res[0] = np.sum(X_b.dot(theta) - y)
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta) - y).dot(X_b[:,i])
            
            # return res * 2 / len(X_b)
            return X_b.T.dot(X_b.dot(theta) - y) * 2.0 / len(X_b)

        def gradient_descent(X_b, y, init_theta, eta, n_iters = 1e4, epsilon=1e-8):
            theta = init_theta
            i_iter = 0

            while i_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient

                if(abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
                
                i_iter += 1
            
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        eta = 0.01

        self._theta = gradient_descent(X_b, Y_train, initial_theta, eta)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_sgd(self, X_train, Y_train, n_iters=5, t0=5, t1=50):
        assert X_train.shape[0] == Y_train.shape[0], "the size of X_train must be equal to the size of Y_train. "
        assert n_iters >= 1

        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2.0

        def sgd(X_b, y, init_theta, n_iters, t0=5, t1=50):
            
            def learning_rate(t):
                return t0 / (t + t1)

            theta = init_theta
            m = len(X_b)

            for cur_iter in range(n_iters):
                indexes = np.random.permutation(m)
                X_b_new = X_b[indexes]
                y_new = y[indexes]

                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(cur_iter * m + i) * gradient

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])

        self._theta = sgd(X_b, Y_train, initial_theta, n_iters, t0, t1)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        assert self.interception_ is not None and self.coef_ is not None, "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), "the feature number of X_predict must be equal to X_train"
        
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, Y_test):
        Y_predict = self.predict(X_test)
        return r2_score(Y_test, Y_predict)

    def __repr__(self):
        return "LinearRegression()"
    