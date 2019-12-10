import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

boston = datasets.load_boston()
X = boston.data
Y = boston.target

print(X.shape)
print(Y.shape)

