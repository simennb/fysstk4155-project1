import functions as fun
import regression_methods as reg
import resampling_methods as res
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import time


def test_bootstrap(fig_path):
    """
    Compares my bootstrap implementation to the example code in lecture notes week 36, slide 29
    Removed the skl.make_pipeline after verifying that the changed version yields the same result.
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

    start_lecture = time.time()
    for degree in range(maxdegree):
        # From lecture slides
        y_pred = np.empty((y_test.shape[0], n_boostraps))
        tot_unique = np.zeros(n_boostraps)
        n_samp = np.zeros(n_boostraps)

        for i in range(n_boostraps):
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

        polydegree[degree] = degree
        error[degree] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
        bias[degree] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
        variance[degree] = np.mean( np.var(y_pred, axis=1, keepdims=True) )

    end_lecture = time.time()

    # Resetting the seed and trying to get to the same point in the seed
    np.random.seed(2018)
    z = np.linspace(-3, 3, n).reshape(-1, 1)
    zz = np.exp(-z ** 2) + 1.5 * np.exp(-(z - 2) ** 2) + np.random.normal(0, 0.1, z.shape)
    z_train, z_test, zz_train, zz_test = train_test_split(z, zz, test_size=0.2)

    start_self = time.time()
    for degree in range(maxdegree):
        # Creating my own design matrix
        X_train = np.zeros((len(x_train), degree+1))
        X_test = np.zeros((len(x_test), degree+1))
        for p in range(0, degree+1):
            X_train[:, p] = x_train[:, 0]**p
            X_test[:, p] = x_test[:, 0]**p

        # Bootstrap from resampling_methods.py
        OLS = reg.OrdinaryLeastSquares()
        bs = res.Bootstrap(X_train, X_test, y_train[:, 0], y_test, OLS, fun.mean_squared_error)
        error_, bias_, var_, error_t = bs.compute(n_boostraps)
        bs_error[degree] = error_
        bs_bias[degree] = bias_
        bs_variance[degree] = var_

    end_self = time.time()

    print('Time elapsed lecture version: %.5f s' % (end_lecture - start_lecture))
    print('Time elapsed selfmade version: %.5f s' % (end_self - start_self))

    fs = 13
    fig, axs = plt.subplots(2)
    fig.suptitle('Comparison between bootstrap implementations', fontsize=fs)

    axs[0].plot(polydegree, error, label='Error')
    axs[0].plot(polydegree, bias, label='Bias')
    axs[0].plot(polydegree, variance, label='Variance')
    axs[0].set_ylabel('Lecture slides', fontsize=fs)

    axs[1].plot(polydegree, bs_error, label='Error')
    axs[1].plot(polydegree, bs_bias, label='Bias')
    axs[1].plot(polydegree, bs_variance, label='Variance')
    axs[1].set_ylabel('Self-made', fontsize=fs)
    axs[1].set_xlabel('Polynomial degree p', fontsize=fs)

    axs[0].legend()
    axs[1].legend()
    axs[0].grid('on')
    axs[1].grid('on')

    plt.savefig(fig_path + 'bootstrap_comparison_N%d_Nbs%d.png' % (n, n_boostraps))
    plt.show()


if __name__ == '__main__':
    fig_path = '../figures/unit_test/'
    test_bootstrap(fig_path)