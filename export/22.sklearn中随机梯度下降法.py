# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # 随机梯度下降法的封装

# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
m = 1000000

x = np.random.normal(size=m)
X = x.reshape(-1, 1)
y = 4. * x + 3. + np.random.normal(0, 3, size=m)


# %%
from playML.LinearRegression import LinearRegression


# %%
lin_reg = LinearRegression()
lin_reg.fit_sgd(X, y, n_iters=2)


# %%
lin_reg.coef_


# %%
lin_reg.interception_

# %% [markdown]
# ## 真实数据应用

# %%
from sklearn import datasets

boston = datasets.load_boston()
X = boston.data
Y = boston.target

X = X[Y < 50.0]
Y = Y[Y < 50.0]


# %%
from playML.train_test_split_self import train_test_split_self
X_train, X_test, Y_train, Y_test = train_test_split_self(X, Y, seed=666)


# %%
from sklearn.preprocessing import StandardScaler
stdScaler = StandardScaler()
stdScaler.fit(X_train)

X_train_standard = stdScaler.transform(X_train)
X_test_standard = stdScaler.transform(X_test)


# %%
lin_reg = LinearRegression()
get_ipython().run_line_magic('time', 'lin_reg.fit_sgd(X_train_standard, Y_train, n_iters=100)')


# %%
lin_reg.score(X_test_standard, Y_test)


# %%
lin_reg2 = LinearRegression()
get_ipython().run_line_magic('time', 'lin_reg2.fit_normal(X_train_standard, Y_train)')


# %%
lin_reg2.score(X_test_standard, Y_test)

# %% [markdown]
# ## scikit-learn中的SGD

# %%
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression


# %%
sgd_reg = SGDRegressor(n_iter_no_change=100)
get_ipython().run_line_magic('time', 'sgd_reg.fit(X_train_standard, Y_train)')
sgd_reg.score(X_test_standard, Y_test)


# %%
lin_reg = LinearRegression()
get_ipython().run_line_magic('time', 'lin_reg.fit(X_train, Y_train)')
lin_reg.score(X_test, Y_test)


# %%



# %%



