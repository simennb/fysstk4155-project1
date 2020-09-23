import numpy as np
import functions as fun

import matplotlib.pyplot as plt


def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


# Making meshgrid of datapoints and compute Franke's function
np.random.seed(4155)
p = 10
n = 13
N = n*n
test_size = 0.2
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)
z_mesh = 1 * np.random.randn(n, n)
x = np.ravel(x)
y = np.ravel(y)
z = np.ravel(z)

trainError = np.zeros(p)
testError = np.zeros(p)
polydegree = range(1,p+1)
for degree in range(1, p+1):
    X = fun.generate_polynomial(x, y, degree)
    #X = create_X(x, y, n=degree)

    X_train, X_test, z_train, z_test = fun.split_data(X, z, test_size=test_size)
    #print(degree, X_train.shape, X_test.shape, z_train.shape, z_test.shape)

    X_train_scaled = fun.scale_X(X_train)
    X_test_scaled = fun.scale_X(X_test)

    beta = np.linalg.pinv(X_train_scaled) @ z_train
    z_fit = X_train_scaled @ beta
    z_pred = X_test_scaled @ beta

    #print(z_fit.shape, z_pred.shape)

    trainError[degree - 1] = fun.mean_squared_error(z_train, z_fit)
    testError[degree - 1] = fun.mean_squared_error(z_test, z_pred)

fun.plot_MSE_SIMPLE(polydegree, trainError, testError, n, test_size)
plt.show()
#fun.plot_MSE_train_test(polydegree, trainError, testError, n_franke, test_size, noise,
#                        fig_path, 'test')
#print(X)
