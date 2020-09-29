# from functions import *
import functions as fun
import regression_methods as reg
import resampling_methods as res
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sklearn.utils import resample
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# TODO: Clean up imports

# TODO: sys.argv would be neat, but not sure how to easily swap in pycharm
print('Input which part of the program to run:')
print('Franke function: a/b/c/d/e')
print('Terrain data: f/g')
run_mode = (input('Run mode: ')).lower()
if run_mode not in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
    sys.exit('Please double check input.')

data = 'franke' if run_mode in ['a', 'b', 'c', 'd', 'e'] else None
data = 'terrain' if run_mode in ['f', 'g'] else data

# Common variables for both parts for easy adjusting
p_dict = {'a': 5, 'b': 10, 'c': 5, 'd': 10,
          'e': 5, 'f': 10, 'g': 5, 'h': 10}
scale_dict = {'a': [True, False], 'b': [True, False], 'c': [True, False], 'd': [True, False],
              'e': [True, False], 'f': [True, False], 'g': [True, False], 'h': [True, False]}
p = p_dict[run_mode]
scale = scale_dict[run_mode]

test_size = 0.2
fig_path = '../figures/'

OLSmethod = 5

if data == 'franke':
    # Creating data set for the Franke function tasks
    np.random.seed(4155)
    n_franke = 13  # number of samples of x/y
    N = n_franke**2  # Total number of samples n*2
    noise = 0.0  # 2
    p = p_dict[run_mode]  # degree of polynomial for the task

    # Randomly generated meshgrid
    x = np.sort(np.random.uniform(0.0, 1.0, n_franke))
    y = np.sort(np.random.uniform(0.0, 1.0, n_franke))

    x_mesh, y_mesh = np.meshgrid(x, y)
    z_mesh = fun.franke_function(x_mesh, y_mesh)

    # Adding normally distributed noise with strength noise
    z_mesh = z_mesh + noise * np.random.randn(n_franke, n_franke)

    # Raveling
    x_ravel = np.ravel(x_mesh)
    y_ravel = np.ravel(y_mesh)
    z_ravel = np.ravel(z_mesh)

    # Creating polynomial design matrix
    X = fun.generate_polynomial(x_ravel, y_ravel, p)

    # Plot franke function if task a for example?


if run_mode == 'a':
    # Splitting into train and test data
    X_train, X_test, z_train, z_test = fun.split_data(X, z_ravel, test_size=test_size)  # TODO: check
#    X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=0.2)

    # Scaling the data
    X_train_scaled = fun.scale_X(X_train, scale)
    X_test_scaled = fun.scale_X(X_test, scale)

    print('Meow ', np.array_equal(X_train_scaled, X_train))

    # Ordinary Least Squares
    OLS = reg.OrdinaryLeastSquares()
    betaOLS = OLS.fit(X_train_scaled, z_train)
    ztildeOLS = OLS.predict(X_test_scaled)

    # Printing MSE and R2 score
    # TODO: maybe save results to file?
    fun.print_MSE_R2(z_test, ztildeOLS, 'test', 'OLS')

    # Confidence interval TODO: fix comments
    conf = OLS.confidence_interval_beta(X_test, z_test, ztildeOLS)
    fun.plot_confidence_int(betaOLS, conf, 'OLS', fig_path, run_mode)
    plt.show()


if run_mode == 'b':
    N_samples = N  # number of samples per bootstrap
    N_bootstraps = 100  # number of resamples

    # Splitting into train and test data
    X_train, X_test, z_train, z_test = fun.split_data(X, z_ravel, test_size=test_size)
#    X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=test_size)

    # Scaling the data
    X_train_scaled = fun.scale_X(X_train, scale)
    X_test_scaled = fun.scale_X(X_test, scale)
#    X_train_scaled = X_train
#    X_test_scaled = X_test

    polydegree = np.arange(1, p + 1)
    trainError = np.zeros(p)
    testError = np.zeros(p)
    bs_mean_OLS = np.zeros(p)
    bs_var_OLS = np.zeros(p)
    bs_bias_OLS = np.zeros(p)

    OLS = reg.OrdinaryLeastSquares(OLSmethod)
    for degree in range(1, p + 1):
        n_poly = fun.polynom_N_terms(degree)
        print('p = %2d, np = %3d' % (degree, n_poly))

        X_train_bs = np.zeros((len(X_train), n_poly))
        X_test_bs = np.zeros((len(X_test), n_poly))
        X_train_OLS = np.zeros((len(X_train), n_poly))
        X_test_OLS = np.zeros((len(X_test), n_poly))

#        X_train_new[:, :] = X_train_scaled[:, 0:n_poly]
#        X_test_new[:, :] = X_test_scaled[:, 0:n_poly]

        X_train_bs[:, :] = X_train[:, 0:n_poly]
        X_test_bs[:, :] = X_test[:, 0:n_poly]

        X_train_OLS[:, :] = X_train_scaled[:, 0:n_poly]
        X_test_OLS[:, :] = X_test_scaled[:, 0:n_poly]

        X_train_OLS = X_train_scaled[:, 0:n_poly]
        X_test_OLS = X_test_scaled[:, 0:n_poly]

        # Ordinary Least Squares without Bootstrapping
        betaOLS = OLS.fit(X_train_OLS, z_train)
        z_trainOLS = OLS.predict(X_train_OLS)
        z_testOLS = OLS.predict(X_test_OLS)

        trainError[degree-1] = fun.mean_squared_error(z_train, z_trainOLS)
        testError[degree-1] = fun.mean_squared_error(z_test, z_testOLS)

        # Bootstrap
        OLS = reg.OrdinaryLeastSquares(OLSmethod)
        bs = res.Bootstrap(X_train_bs, X_test_bs, z_train, z_test, OLS, fun.mean_squared_error)
        mean_OLS, var_OLS, bias_OLS = bs.compute(N_bootstraps)
        bs_mean_OLS[degree-1] = mean_OLS
        bs_var_OLS[degree-1] = var_OLS
#        print(bias_OLS, type(bias_OLS))
        bs_bias_OLS[degree-1] = bias_OLS
#        print(mean_OLS, var_OLS, bias_OLS)

        '''
        for i in range(N_bootstraps):
            X_train_BS, z_train_BS = resample(X_train_new, z_train)
            # Scaling the data
            X_train_BS = fun.scale_X(X_train_BS)
            if degree == 10 and i == 33:
                print(z_train_BS)

            betaBS = OLS.fit(X_train_BS, z_train_BS)
            z_fitBS = OLS.predict(X_train_BS)
            z_predBS = OLS.predict(X_test_new)

            a = (z_test - np.mean(z_predBS)) ** 2
#            print(a)
            if degree == 10 and i == 33:
                print(i, z_predBS)
                print(i, z_fitBS)
                print(np.mean((z_train_BS - z_fitBS)**2))
            bs_mean_OLS[degree-1] += np.mean((z_test - z_predBS) ** 2) / N_bootstraps
            bs_var_OLS[degree-1] += np.var(z_predBS) / len(z_predBS)
            bs_bias_OLS[degree-1] += np.mean((z_test - np.mean(z_predBS)) ** 2) / N_bootstraps
            '''
    '''
    fig = plt.figure()
    plt.plot(polydegree, trainError, label='Train')
    plt.plot(polydegree, testError, label='Test')
    plt.plot(polydegree, bs_mean_OLS, label='BS Test')
    plt.grid('on')
    plt.legend()
    #    plt.yscale('log')
    plt.show()
    '''

#    polydegree, train_MSE, test_MSE, n_a, test_size, noise, fig_path, task, resample = None)
    fun.plot_MSE_train_test(polydegree, trainError, bs_mean_OLS, n_franke, test_size, noise,
                            fig_path, run_mode, 'Bootstrap')

    fun.plot_MSE_train_test(polydegree, trainError, testError, n_franke, test_size, noise,
                            fig_path, run_mode)

#    fun.plot_MSE_test_OLS_fit(polydegree, trainError, testError, n_franke, test_size, noise, OLSmethod)

    fun.plot_bias_variance(polydegree, bs_mean_OLS, bs_bias_OLS, bs_var_OLS, 'trade-off', fig_path, run_mode)

    plt.show()


if run_mode == 'c':
    K = 5

    polydegree = np.arange(1, p + 1)
    error_CV = np.zeros(len(polydegree))
    error_SKL = np.zeros(len(polydegree))

    X_scaled = fun.scale_X(X, scale)
    OLS = reg.OrdinaryLeastSquares(OLSmethod)
    for degree in range(1, p + 1):
        n_poly = fun.polynom_N_terms(degree)
        print('p = %2d, np = %3d' % (degree, n_poly))

        X_cv = np.zeros((len(X_scaled), n_poly))
        X_cv[:, :] = X_scaled[:, 0:n_poly]

        CV = res.CrossValidation(X_cv, z_ravel, OLS, fun.mean_squared_error)
        error_CV[degree-1] = CV.compute(K)

        print('K-fold cross-validation:')
        print('K=%d, MSE = %.5f' % (K, error_CV[degree-1]))

        kfold = KFold(n_splits=K)
    #    kfold = KFold(n_splits=K, random_state=None, shuffle=True)
        for train_inds, test_inds in kfold.split(X):
            X_train = X_cv[train_inds]
            z_train = z_ravel[train_inds]

            X_test = X_cv[test_inds]
            z_test = z_ravel[test_inds]

            betaOLS = OLS.fit(X_train, z_train)
            z_testOLS = OLS.predict(X_test)

            error_SKL[degree-1] += fun.mean_squared_error(z_test, z_testOLS) / K

        print('K-fold cross-validation SKL:')
        print('K=%d, MSE = %.5f' % (K, error_SKL[degree-1]))

    fun.plot_multiple_y(polydegree, [error_CV, error_SKL], ['CV', 'SKL'], 'Comparing different CV implementations',
                        'Polynomial degree', 'Mean squared error', 'compare_CV_SKL_p%d' % p, fig_path, run_mode)
    plt.show()


if run_mode == 'd':
    # Setting up for Ridge regression
    nlambdas = 100
    lambdas = np.logspace(-4, 1, nlambdas)

    # Bootstrap
    N_bootstraps = 100

    # Cross-validation
    K = 5

    # Splitting into train and test data
    X_train, X_test, z_train, z_test = fun.split_data(X, z_ravel, test_size=test_size)
    #    X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=test_size)

    # Scaling the data
    X_train_scaled = fun.scale_X(X_train, scale)
    X_test_scaled = fun.scale_X(X_test, scale)
    #    X_train_scaled = X_train
    #    X_test_scaled = X_test
    X_scaled = fun.scale_X(X)

    polydegree = np.arange(1, p + 1)

    # Bootstrap arrays
    bs_error_Ridge = np.zeros((p, nlambdas))
    bs_var_Ridge = np.zeros((p, nlambdas))
    bs_bias_Ridge = np.zeros((p, nlambdas))

    # error/var/bias at the lambda that gives minimum
    # TODO: Though maybe a common value for them is correct to do?
    bs_error_optimal = np.zeros(p)
    bs_var_optimal = np.zeros(p)
    bs_bias_optimal = np.zeros(p)
    bs_lmb_optimal = np.zeros(p)

    # Cross-validation arrays
    cv_error_Ridge = np.zeros((p, nlambdas))
    cv_error_optimal = np.zeros(p)
    cv_lmb_optimal = np.zeros(p)
#    test_Ridge = np.zeros((p, nlambdas))

    Ridge = reg.RidgeRegression()
    for degree in range(1, p + 1):
        n_poly = fun.polynom_N_terms(degree)
        print('p = %2d, np = %3d' % (degree, n_poly))

        X_train_bs = np.zeros((len(X_train_scaled), n_poly))
        X_test_bs = np.zeros((len(X_test_scaled), n_poly))
        X_cv = np.zeros((len(X_scaled), n_poly))

        X_train_bs[:, :] = X_train_scaled[:, 0:n_poly]
        X_test_bs[:, :] = X_test_scaled[:, 0:n_poly]
        X_cv[:, :] = X_scaled[:, 0:n_poly]

        for i in range(nlambdas):
            lmb = lambdas[i]
            Ridge.set_lambda(lmb)

            if i % 10 == 0:
                print('i = %d, lmb= %.3f' % (i, lmb))

            # Bootstrap
            BS = res.Bootstrap(X_train_bs, X_test_bs, z_train, z_test, Ridge, fun.mean_squared_error)
            error_, var_, bias_ = BS.compute(N_bootstraps)
            bs_error_Ridge[degree-1, i] = error_
            bs_var_Ridge[degree-1, i] = var_
            bs_bias_Ridge[degree-1, i] = bias_

            # Cross validation
            CV = res.CrossValidation(X_cv, z_ravel, Ridge, fun.mean_squared_error)
            cv_error_Ridge[degree-1, i] = CV.compute(K)


#        bs_error_optimal[degree-1] = np.min(bs_error_Ridge[degree-1, :])
#        bs_var_optimal[degree-1] = np.min(bs_var_Ridge[degree-1, :])
#        bs_bias_optimal[degree-1] = np.min(bs_bias_Ridge[degree-1, :])

        # Bootstrap
        index_bs = np.argmin(bs_error_Ridge[degree-1, :])
        bs_lmb_optimal[degree-1] = lambdas[index_bs]
        bs_error_optimal[degree-1] = bs_error_Ridge[degree-1, index_bs]
        bs_var_optimal[degree-1] = bs_var_Ridge[degree-1, index_bs]
        bs_bias_optimal[degree-1] = bs_bias_Ridge[degree-1, index_bs]

        # Cross-validation
        index_cv = np.argmin(cv_error_Ridge[degree-1, :])
        cv_lmb_optimal[degree-1] = lambdas[index_cv]
        cv_error_optimal[degree-1] = cv_error_Ridge[degree-1, index_cv]

    # Bootstrap Plots
    fun.plot_bias_variance(polydegree, bs_error_optimal, bs_bias_optimal, bs_var_optimal,
                           'Ridge regression', fig_path, run_mode)
    fun.plot_degree_lambda(polydegree, bs_lmb_optimal, 'Bootstrap $\lambda$ value at min(error)',
                           'bootstrap', fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, bs_error_Ridge, 'Bootstrap + Ridge, MSE',
                     'bs_ridge_error', fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, bs_bias_Ridge, 'Bootstrap + Ridge, bias',
                     'bs_ridge_bias', fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, bs_var_Ridge, 'Bootstrap + Ridge, variance',
                     'bs_ridge_variance', fig_path, run_mode)

    # Cross-Validation Plots
    fun.plot_degree_lambda(polydegree, cv_lmb_optimal, 'CV $\lambda$ value at min(error)',
                           'cv', fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, cv_error_Ridge, 'CV + Ridge, MSE',
                     'cv_ridge_error', fig_path, run_mode)


    '''
    fig = plt.figure()
#    p_mesh, lmb_mesh = np.meshgrid(polydegree, lambdas)
    heatmap = plt.pcolor(lambdas, polydegree, bs_error_Ridge)
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Polynomial degree')
    fig.colorbar(heatmap)#bs_error_Ridge)
#    plt.pcolor(bs_error_Ridge)
    '''
    plt.show()


if run_mode == 'e':
    pass


if data == 'terrain':
    pass

if run_mode == 'f':
    pass
if run_mode == 'g':
    pass
