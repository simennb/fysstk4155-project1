import numpy as np
from functions import *

class OrdinaryLeastSquares:
    def __init__(self):
        self.beta = None
        self.ytilde = None

    def fit(self, X, y):
        self.beta = invert_SVD(X.T @ X) @ X.T @ y  # all work now
#        self.beta = SVDinv(X.T @ X) @ X.T @ y  #
#        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y  #
#        self.beta = np.linalg.pinv(X.T @ X) @ X.T @ y  #
#        self.beta = np.linalg.pinv(X) @ y
        return self.beta

    def predict(self, X):
        ytilde = X @ self.beta
        return ytilde


class RidgeRegression:
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass