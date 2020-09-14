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

if data == 'franke':
    np.random.seed(4155)
    n_a = 100  # number of points for task a
    p = 5  # degree of polynomial

    # Make meshgrid for the Franke function
#    x = np.arange(0, 1, 0.01)
#    y = np.arange(0, 1, 0.01)

    # Randomly generated meshgrid, TODO: test both maybe?
    x = np.sort(np.random.uniform(0.0, 1.0, n_a))
    y = np.sort(np.random.uniform(0.0, 1.0, n_a))

    x_mesh, y_mesh = np.meshgrid(x, y)

    z_mesh = franke_function(x_mesh, y_mesh)

    # Adding random noise
#    z_mesh = z_mesh + 0.025 * np.random.randn(n_a, n_a)

    # Ravel!? YES
    x_ravel = np.ravel(x_mesh)
    y_ravel = np.ravel(y_mesh)
    z_ravel = np.ravel(z_mesh)

    # Creating polynomial design matrix
    X = generate_polynomial(x_ravel, y_ravel, p)


if run_mode == 'a' or run_mode == 'all_f':
    # Splitting into train and test data
    X_train, X_test, z_train, z_test = split_data(X, z_ravel, test_size=0.2)
    #    X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=0.2)
    # TODO: check
#    X_train, X_test, z_train, z_test = X, X, z_ravel, z_ravel

    # Scaling the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    #X_train_scaled = scaler.transform(X_train)
    #X_test_scaled = scaler.transform(X_test)
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
    X_train, X_test, z_train, z_test = split_data(X, z_ravel, test_size=0.2)
    #    X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=0.2)
    # TODO: check

    # Scaling the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    polydegree = np.arange(1, p + 1)
    trainError = np.zeros(len(range(1, p + 1)))
    testError = np.zeros(len(range(1, p + 1)))
    for degree in range(1, p + 1):
        n_poly = polynom_N_terms(degree)
        print('p = ', degree, n_poly)
        X_train_new = X_train_scaled[:, 0:n_poly]
        X_test_new = X_test_scaled[:, 0:n_poly]


        # Bootstrap
        OLS = reg.OrdinaryLeastSquares()
        bs = res.Bootstrap(X_train_new, X_test_new, z_train, z_test, OLS, [mean_squared_error])
        mean_OLS, var_OLS = bs.compute(100, 10)
        print(mean_OLS, var_OLS)


        OLS = reg.OrdinaryLeastSquares()
        betaOLS = OLS.fit(X_train_new, z_train)
        z_trainOLS = OLS.predict(X_train_new)
        z_testOLS = OLS.predict(X_test_new)

        #######
        trainError[degree-1] = mean_squared_error(z_train, z_trainOLS)
        testError[degree-1] = mean_squared_error(z_test, z_testOLS)

    fig = plt.figure()
    plt.plot(polydegree, trainError)
    plt.plot(polydegree, testError)
    plt.show()


if run_mode == 'c' or run_mode == 'all_f':
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
