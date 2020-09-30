import numpy as np
import functions as fun


class OrdinaryLeastSquares:
    def __init__(self, method=5):
        self.beta = None
        self.ytilde = None
        self.method = method  # for testing fit

    def fit(self, X, y):

        if self.method == 1:
            self.beta = fun.invert_SVD(X.T @ X) @ X.T @ y  # Method 1

        elif self.method == 2:
            self.beta = fun.SVDinv(X.T @ X) @ X.T @ y  # Method 2

        elif self.method == 3:
            self.beta = np.linalg.inv(X.T @ X) @ X.T @ y  # Method 3

        elif self.method == 4:
            self.beta = np.linalg.pinv(X.T @ X) @ X.T @ y  # Method 4

        elif self.method == 5:
            self.beta = np.linalg.pinv(X) @ y  # Method 5

        return self.beta

    def predict(self, X):
        ytilde = X @ self.beta
        return ytilde

    def confidence_interval_beta(self, X, y, ytilde):
        N = X.shape[0]
        p = X.shape[1]

        sigma2 = 1/(N-p-1) * np.sum((y - ytilde)**2)
        var_beta = np.diagonal(np.linalg.pinv(X.T @ X)) * sigma2

        print(sigma2, var_beta)
        conf = 2*np.sqrt(var_beta)

        # Mathias
        '''
        N, P = X.shape
        z_variance = np.sum((z.ravel() - z_predict_)**2) / (N - P - 1)

        linreg_coef_var = np.diag(np.linalg.inv(X.T @ X))*z_variance
        np.sqrt(linreg_coef_var) * 2,
        '''
        return conf

# Eirik
'''
def ConfIntBeta(self, X, zr, pred):
    N = X.shape[0]
    p = X.shape[1]

    variance = 1. / (N - p - 1) * np.sum((zr - pred) ** 2)

    self.var_beta = [(np.linalg.inv(X.T @ X))[i, i] for i in range(0, p)]

    self.conf_intervals = [[float(self.beta[i]) - 2 * np.sqrt(self.var_beta[i]),
                            float(self.beta[i]) + 2 * np.sqrt(self.var_beta[i])] for i in range(0, len(self.var_beta))]

    return self
'''

class RidgeRegression:
    def __init__(self):
        self.beta = None
        self.ytilde = None
        self.lmb = None

    def set_lambda(self, lmb):
        self.lmb = lmb

    def fit(self, X, y):
        I = np.identity(X.shape[1])
        self.beta = np.linalg.pinv(X.T @ X + self.lmb * I) @ X.T @ y

    def predict(self, X):
        ytilde = X @ self.beta
        return ytilde

    def confidence_interval_beta(self, X, y, ytilde):
        # TODO: add lambda dependency
        N = X.shape[0]
        p = X.shape[1]

        sigma2 = 1/(N-p-1)*np.sum((y-ytilde)**2)
        var_beta = np.diagonal(np.linalg.pinv(X.T @ X)) * sigma2
        conf = 2*np.sqrt(var_beta)

        return conf
