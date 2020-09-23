import functions as fun
import regression_methods as reg
import resampling_methods as res

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample


def test_bootstrap():
    """
    Compares my bootstrap implementation to the example code in lecture notes week 36, slide 29
    Removed the skl.make_pipeline after verifying that the changed version yields the same result.
    :return:
    """
    np.random.seed(2018)

    n = 40
    n_boostraps = 100
    maxdegree = 14

    # Make data set.
    x = np.linspace(-3, 3, n).reshape(-1, 1)
    y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)
    error = np.zeros(maxdegree)
    bias = np.zeros(maxdegree)
    variance = np.zeros(maxdegree)
    polydegree = np.zeros(maxdegree)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Self made arrays
    bs_error = np.zeros(maxdegree)
    bs_bias = np.zeros(maxdegree)
    bs_variance = np.zeros(maxdegree)

    for degree in range(maxdegree):
        X_train = np.zeros((len(x_train), degree+1))
        X_test = np.zeros((len(x_test), degree+1))
        for p in range(0, degree+1):
            X_train[:, p] = x_train[:, 0]**p
            X_test[:, p] = x_test[:, 0]**p

        # From lecture slides
        y_pred = np.empty((y_test.shape[0], n_boostraps))
        tot_unique = np.zeros(n_boostraps)
        n_samp = np.zeros(n_boostraps)
        for i in range(n_boostraps):
            # [8 2 7 6 0 2 9 1 1 3]
            x_, y_ = resample(x_train, y_train)
            tot_unique[i] = len(np.unique(x_))
            n_samp[i] = len(x_)

            X_train = np.zeros((len(x_), degree + 1))
            X_test = np.zeros((len(x_test), degree + 1))
            for p in range(0, degree + 1):
                X_train[:, p] = x_[:, 0] ** p
                X_test[:, p] = x_test[:, 0] ** p

            beta_ols = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_
            y_pred[:, i] = (X_test @ beta_ols)[:, 0]

        print('N_samples = %.2f , mean(unique) = %.2f' % (np.mean(n_samp), np.mean(tot_unique)))
#        print('Avg. number of unique in lecture bootstrap: %.2f' % np.mean(tot_unique))

        polydegree[degree] = degree
        error[degree] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
        bias[degree] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
        variance[degree] = np.mean( np.var(y_pred, axis=1, keepdims=True) )

    # Resetting the seed and trying to get to the same point in the seed
    np.random.seed(2018)
    z = np.linspace(-3, 3, n).reshape(-1, 1)
    zz = np.exp(-z ** 2) + 1.5 * np.exp(-(z - 2) ** 2) + np.random.normal(0, 0.1, z.shape)
    z_train, z_test, zz_train, zz_test = train_test_split(z, zz, test_size=0.2)

    for degree in range(maxdegree):
        # Creating my own design matrix
        X_train = np.zeros((len(x_train), degree+1))
        X_test = np.zeros((len(x_test), degree+1))
        for p in range(0, degree+1):
            X_train[:, p] = x_train[:, 0]**p
            X_test[:, p] = x_test[:, 0]**p

        # Bootstrap from resampling_methods.py
        # [8 2 7 6 0 2 9 1 1 3]
        OLS = reg.OrdinaryLeastSquares()
        bs = res.Bootstrap(X_train, X_test, y_train[:, 0], y_test, OLS, fun.mean_squared_error)
        mean_TEST, var_TEST, bias_TEST = bs.compute(n, n_boostraps)
        bs_error[degree] = mean_TEST
        bs_bias[degree] = bias_TEST
        bs_variance[degree] = var_TEST
        #        print(bias_OLS, type(bias_OLS))
        #        print(mean_OLS, var_OLS, bias_OLS)


        # Compare printout
        print('Polynomial degree:', degree)
        '''
        print('Error: %.7f , Self: %.7f' % (error[degree], bs_error[degree]))
        print('Bias^2: %.7f , Self: %.7f' % (bias[degree], bs_bias[degree]))
        print('Var: %.7f , Self: %.7f' % (variance[degree], bs_variance[degree]))
#        print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))
        '''
    fig, axs = plt.subplots(2)
    fig.suptitle('Comparison between bootstrap implementations')

    axs[0].plot(polydegree, error, label='Error')
    axs[0].plot(polydegree, bias, label='Bias')
    axs[0].plot(polydegree, variance, label='Variance')
    axs[0].set_ylabel('Lecture slides')

    axs[1].plot(polydegree, bs_error, label='BS Error')
    axs[1].plot(polydegree, bs_bias, label='BS Bias')
    axs[1].plot(polydegree, bs_variance, label='BS Variance')
    axs[1].set_ylabel('Self')

    axs[0].legend()
    axs[1].legend()
    axs[0].grid('on')
    axs[1].grid('on')
    plt.show()

    '''
    plt.figure()
    plt.plot(polydegree, error, label='Error')
    plt.plot(polydegree, bias, label='bias')
    plt.plot(polydegree, variance, label='Variance')
    plt.legend()

    plt.figure()
    plt.plot(polydegree, bs_error, label='BS Error')
    plt.plot(polydegree, bs_bias, label='BS bias')
    plt.plot(polydegree, bs_variance, label='BS Variance')
    plt.legend()
    '''

    plt.show()

if __name__ == '__main__':
    test_bootstrap()