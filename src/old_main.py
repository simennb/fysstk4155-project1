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
from sklearn.preprocessing import StandardScaler

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# TODO: Clean up imports

# TODO: sys.argv would be neat, but not sure how to easily swap in pycharm
print('Input which part of the program to run:')
print('Franke function: a/b/c/d/e/all_f')
print('Terrain data: f/g/all_t')
run_mode = (input('Run mode: ')).lower()
if run_mode not in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'all_f', 'all_t']:
    sys.exit('Please double check input.')

data = 'franke' if run_mode in ['a', 'b', 'c', 'd', 'e', 'all_f'] else None
data = 'terrain' if run_mode in ['f', 'g', 'all_t'] else data

# Common variables for both parts for easy adjusting
p_dict = {'a': 5, 'b': 10}  # TODO: kinda problematic for all_f/all_t
test_size = 0.2
fig_path = '../figures/'

OLSmethod = 5

if data == 'franke':
    # Creating data set for the Franke function tasks
    np.random.seed(4155)
    n_franke = 13  # number of samples of x/y, so total N is n_franke^2
    N = n_franke**2
    noise = 0.0  # 2
    p = p_dict[run_mode]  # degree of polynomial for the task

    # Randomly generated meshgrid
    # TODO: do i randomly draw samples and make meshgrid, or n_f*n_f random sets of (x,y) samples?
    x = np.sort(np.random.uniform(0.0, 1.0, n_franke))
    y = np.sort(np.random.uniform(0.0, 1.0, n_franke))
    # TODO: is sort even necessary when im not plotting it?

#    x = np.random.uniform(0.0, 1.0, n_franke)
#    y = np.random.uniform(0.0, 1.0, n_franke)

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

    # Alternative approach with n^2 (x,y) pairs
    '''
    x_alt = np.random.uniform(0.0, 1.0, n_franke*n_franke)
    y_alt = np.random.uniform(0.0, 1.0, n_franke*n_franke)
    a = np.random.randn(5)
    # TODO: drawing 0-4 random numbers give R2 score of ~0.3-0.4
#    z_ravel = fun.franke_function(x_alt, y_alt)
#    X = fun.generate_polynomial(x_alt, y_alt, p)
    '''

    # Plot franke function if task a for example?


if run_mode == 'a' or run_mode == 'all_f':
    # Splitting into train and test data
    X_train, X_test, z_train, z_test = fun.split_data(X, z_ravel, test_size=test_size)  # TODO: check
#    X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=0.2)

    # Scaling the data
    X_train_scaled = fun.scale_X(X_train)
    X_test_scaled = fun.scale_X(X_test)

    # Ordinary Least Squares
    OLS = reg.OrdinaryLeastSquares()
    betaOLS = OLS.fit(X_train_scaled, z_train)
    ztildeOLS = OLS.predict(X_test_scaled)

    # Printing MSE and R2 score
    # TODO: maybe save results to file?
    fun.print_MSE_R2(z_test, ztildeOLS, 'test', 'OLS')


if run_mode == 'b' or run_mode == 'all_f':
#    N_bs = 169  # number of samples per bootstrap
#    N_resamples = 10  # number of resamples

    # Splitting into train and test data
    X_train, X_test, z_train, z_test = fun.split_data(X, z_ravel, test_size=test_size)
#    X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=test_size)

    # Scaling the data
    X_train_scaled = fun.scale_X(X_train)
    X_test_scaled = fun.scale_X(X_test)
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
        print('p = %2d, np = %3d' %(degree, n_poly))
        X_train_new = X_train_scaled[:, 0:n_poly]
        X_test_new = X_test_scaled[:, 0:n_poly]

        # Bootstrap
        '''
        OLS = reg.OrdinaryLeastSquares()
        bs = res.Bootstrap(X_train_new, X_test_new, z_train, z_test, OLS, fun.mean_squared_error)
        mean_OLS, var_OLS, bias_OLS = bs.compute(N_bs, N_resamples)
        bs_mean_OLS[degree-1] = mean_OLS
        bs_var_OLS[degree-1] = var_OLS
#        print(bias_OLS, type(bias_OLS))
        bs_bias_OLS[degree-1] = bias_OLS
#        print(mean_OLS, var_OLS, bias_OLS)
        '''

        # Test
        betaOLS = OLS.fit(X_train_new, z_train)
        z_trainOLS = OLS.predict(X_train_new)
        z_testOLS = OLS.predict(X_test_new)

        #######
        trainError[degree-1] = fun.mean_squared_error(z_train, z_trainOLS)
        testError[degree-1] = fun.mean_squared_error(z_test, z_testOLS)

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
#    fun.plot_MSE_train_test(polydegree, trainError, bs_mean_OLS, n_franke, test_size, noise,
#                            fig_path, run_mode, 'Bootstrap')

    fun.plot_MSE_train_test(polydegree, trainError, testError, n_franke, test_size, noise,
                            fig_path, run_mode)

#    fun.plot_MSE_test_OLS_fit(polydegree, trainError, testError, n_franke, test_size, noise, OLSmethod)

#    fun.plot_bias_variance(polydegree, bs_mean_OLS, bs_bias_OLS, bs_var_OLS)

    plt.show()


if run_mode == 'c' or run_mode == 'all_f':
    # TODO 1c dont do bias variance trade off with cross validation unless scikit learn since it takes more work to do by hand
    # TODO DONT USE TRAIN TEST SPLIT, cross validation does it
    pass
if run_mode == 'd' or run_mode == 'all_f':
    pass
if run_mode == 'e' or run_mode == 'all_f':
    pass


if data == 'terrain':
    pass

if run_mode == 'f' or run_mode == 'all_t':
    pass
if run_mode == 'g' or run_mode == 'all_t':
    pass