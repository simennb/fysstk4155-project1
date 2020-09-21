from functions import *
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


# TODO: sys.argv would be neat, but not sure how to easily swap in pycharm
# Creating different run modes in order to easily swap between the different tasks
# Or add an ALL mode to run everything?
# Probably just swap between franke and real data
print('Input which part of the program to run:')
print('Franke function: a/b/c/d/e/all_f')
print('Terrain data: f/g/all_t')
run_mode = (input('Run mode: ')).lower()
if run_mode not in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'all_f', 'all_t']:
    sys.exit('Please double check input.')

data = 'franke' if run_mode in ['a', 'b', 'c', 'd', 'e', 'all_f'] else None
data = 'terrain' if run_mode in ['f', 'g', 'all_t'] else data

#run_mode = (input('franke/terrain/all? ')).lower()
#if run_mode not in ['franke', 'terrain', 'all']:
#    sys.exit('Please double check input.')

# Some common variables
p_dict = {'a': 5, 'b': 20}
test_size = 0.2
noise = 0.0#2


if data == 'franke':
    np.random.seed(4155)
    # TODO: Make n_a and p depend on task?
    # TODO n_a is the number of x/y samples, so number of data points is n_a^2 !!!!!!!!!!!!!!!!!!!!!!!!!!!
    n_franke = 100  # number of points for task a
    p = p_dict[run_mode]  # todo????

    # Randomly generated meshgrid, TODO: test both maybe?
    x = np.sort(np.random.uniform(0.0, 1.0, n_franke))
    y = np.sort(np.random.uniform(0.0, 1.0, n_franke))

    x_mesh, y_mesh = np.meshgrid(x, y)

    z_mesh = franke_function(x_mesh, y_mesh)

    # Adding random noise
    z_mesh = z_mesh + noise * np.random.randn(n_franke, n_franke)

    # Ravel!? YES
    x_ravel = np.ravel(x_mesh)
    y_ravel = np.ravel(y_mesh)
    z_ravel = np.ravel(z_mesh)

    # Creating polynomial design matrix
    X = generate_polynomial(x_ravel, y_ravel, p)


if run_mode == 'a' or run_mode == 'all_f':
    # Splitting into train and test data
    X_train, X_test, z_train, z_test = split_data(X, z_ravel, test_size=test_size)
    #X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=0.2)
    # TODO: check
#    X_train, X_test, z_train, z_test = X, X, z_ravel, z_ravel

    # Scaling the data
    scaler = StandardScaler()
    scaler.fit(X_train)
#    X_train_scaled = scaler.transform(X_train)
#    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = X_train  # TODO ONLY FOR TEST PURPOSES
    X_test_scaled = X_test


    # Ordinary Least Squares
    OLS = reg.OrdinaryLeastSquares()
    betaOLS = OLS.fit(X_train_scaled, z_train)
    ztildeOLS = OLS.predict(X_test_scaled)

    #betaOLS = OLS.fit(X_train, z_train)
    #ztildeOLS = OLS.predict(X_test)

    # R2, MSE
    # TODO: Print both properly, make function
    print_MSE_R2(z_test, ztildeOLS, 'test', 'OLS')


if run_mode == 'b' or run_mode == 'all_f':
    # Splitting into train and test data
    X_train, X_test, z_train, z_test = split_data(X, z_ravel, test_size=test_size)
    #X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=test_size)
#    X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=test_size, shuffle=False)
    # TODO: check
    print(X_train.shape, X_test.shape, z_train.shape, z_test.shape)
    '''
    X_train = X_train[::n_a]
    X_test = X_test[::n_a]
    z_train = z_train[::n_a]
    z_test = z_test[::n_a]
    print(X_train.shape, X_test.shape, z_train.shape, z_test.shape)
    # (7999, 231) (2001, 231) (7999,) (2001,) mine
    # (8000, 231) (2000, 231) (8000,) (2000,) train test split
    '''
    # Scaling the data
    #scaler = StandardScaler()
    #scaler.fit(X_train)
    #X_train_scaled = scaler.transform(X_train)
    #X_test_scaled = scaler.transform(X_test)

    #X_train_scaled = X_train
    #X_test_scaled = X_test

    polydegree = np.arange(1, p + 1)
    trainError = np.zeros(p)
    testError = np.zeros(p)
    bs_mean_OLS = np.zeros(p)
    bs_var_OLS = np.zeros(p)
    for degree in range(1, p + 1):
        n_poly = polynom_N_terms(degree)
#        print('p = ', degree, n_poly)
#        X_train_new = X_train_scaled[:, 0:n_poly]
#        X_test_new = X_test_scaled[:, 0:n_poly]

        # TODO Fix names
        X_train_red = X_train[:, 0:n_poly]
        X_test_red = X_test[:, 0:n_poly]


#        scaler = StandardScaler()
#        scaler.fit(X_train_red)
#        X_train_scaled = scaler.transform(X_train_red)
#        X_test_scaled = scaler.transform(X_test_red)
#        X_train_new = X_train_scaled
#        X_test_new = X_test_scaled
        # TODO FINALLY
        X_train_new = scale_X(X_train_red)
        X_test_new = scale_X(X_test_red)


        # Bootstrap
        '''
        OLS = reg.OrdinaryLeastSquares()
        bs = res.Bootstrap(X_train_new, X_test_new, z_train, z_test, OLS, [mean_squared_error])
        mean_OLS, var_OLS = bs.compute(100, 10)
        bs_mean_OLS[degree-1] = mean_OLS
        bs_var_OLS[degree-1] = var_OLS
        print(mean_OLS, var_OLS)
        '''

        # TESTING
        beta_optimal = np.linalg.pinv(X_train_new) @ z_train

        print(beta_optimal.shape)
        z_trainOLS = X_train_new @ beta_optimal
        z_testOLS = X_test_new @ beta_optimal


        OLS = reg.OrdinaryLeastSquares()
        betaOLS = OLS.fit(X_train_new, z_train)
#        z_trainOLS = OLS.predict(X_train_new)
#        z_testOLS = OLS.predict(X_test_new)

        #######
        trainError[degree-1] = mean_squared_error(z_train, z_trainOLS)
        testError[degree-1] = mean_squared_error(z_test, z_testOLS)

    plot_MSE_train_test(polydegree, trainError, testError, n_franke, test_size, noise)

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
