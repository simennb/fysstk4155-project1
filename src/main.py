from functions import *
import regression_methods as reg
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Creating different run modes in order to easily swap between the different tasks
# Or add an ALL mode to run everything?
run_mode = (input('a/b/c/d/e/f/g/all? ')).lower()
if run_mode not in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'all']:
    sys.exit('Please double check input.')


if run_mode == 'a' or run_mode == 'all':
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
    z_mesh = z_mesh + 0.025 * np.random.randn(n_a, n_a)

    # Ravel!? YES
    x_ravel = np.ravel(x_mesh)
    y_ravel = np.ravel(y_mesh)
    z_ravel = np.ravel(z_mesh)

    # Creating polynomial design matrix
    X = generate_polynomial(x_ravel, y_ravel, p)

    # Scaling the data
    # TODO

    # Splitting into train and test data
    X_train, X_test, z_train, z_test = split_data(X, z_ravel, test_size=0.2)
    # TODO

    # Ordinary Least Squares
    OLS = reg.OrdinaryLeastSquares()
    betaOLS = OLS.fit(X, z_ravel)
    ztildeOLS = OLS.predict(X)

    print(mean_squared_error(z_ravel, ztildeOLS))
    print(betaOLS)



if run_mode == 'b' or run_mode == 'all':
    pass
if run_mode == 'c' or run_mode == 'all':
    pass
if run_mode == 'd' or run_mode == 'all':
    pass
if run_mode == 'e' or run_mode == 'all':
    pass
if run_mode == 'f' or run_mode == 'all':
    pass
if run_mode == 'g' or run_mode == 'all':
    pass
