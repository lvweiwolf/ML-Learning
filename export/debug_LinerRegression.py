import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

boston = datasets.load_boston()
X = boston.data
Y = boston.target

print (boston.DESCR)

plot_X = X[:, 5]
plot_Y = X[:, 6]
plot_Z = Y

ax = plt.axes(projection='3d')
ax.scatter3D(plot_X, plot_Y, plot_Z, c=plot_Z, cmap='Greens')
plt.show()

print(X.shape)
print(Y.shape)

print(X[:50])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
score = lin_reg.score(X_test, Y_test)
print(score)



