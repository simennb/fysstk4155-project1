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
import sklearn.linear_model as skl
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

data = ''
data = 'franke' if run_mode in ['a', 'b', 'c', 'd', 'e'] else None
data = 'terrain' if run_mode in ['f', 'g'] else data

# Common variables for both parts for easy adjusting
p_dict = {'a': 5, 'b': 15, 'c': 5, 'd': 20,
          'e': 10, 'f': 1, 'g': 15}
scale_dict = {'a': [True, False], 'b': [True, False], 'c': [True, False], 'd': [True, False],
              'e': [True, False], 'f': None, 'g': [True, False]}
p = p_dict[run_mode]  # degree of polynomial for the task
scale = scale_dict[run_mode]

test_size = 0.2
fig_path = '../figures/'
data_path = '../datafiles/'

#reg_g = 'OLS'
reg_g = 'Ridge'
#reg_g = 'Lasso'

OLSmethod = 5

if data == 'franke':
    # Creating data set for the Franke function tasks
    seed = 4155
    np.random.seed(seed)
#    n_franke = 32  # number of samples of x/y
    n_franke = 23  # 529 points
    N = n_franke**2  # Total number of samples n*2
    noise = 0.05  # 2
#    noise = 0.1  # 2
#    noise = 0.8571

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

    # Printing some information for logging purposes
    fun.print_parameters_franke(seed, N, noise, p, scale, test_size)


if data == 'terrain':
    terrain_data = 'SRTM_data_Norway_3.tif'
    # (3601, 1801) dimensions of image
    # Maybe make a size_x, size_y
    # [start_x, start_y, size]
    patches = {1: [880, 2920, 150]}
    patch = 1

    n_terrain = patches[patch][2]
    N = n_terrain**2

#    loc_s = [1550, 0]  # pretty interesting shape
#    loc_s = [880, 2920]  # where i grew up (in _3 map) (n_terrain = 150)

    loc_s = patches[patch][0:2]
    loc_e = [loc_s[0] + n_terrain, loc_s[1] + n_terrain]

    # Fetching x, y, z from map
    x_mesh, y_mesh, z_mesh = fun.read_terrain(data_path + terrain_data, n_terrain, loc_s)

    # Raveling
    x = np.ravel(x_mesh)
    y = np.ravel(y_mesh)
    z = np.ravel(z_mesh)

    # Creating design matrix
    X = fun.generate_polynomial(x, y, p)

    # Printing some information for logging purposes
    print('Terrain datafile: %s' % terrain_data)
    print('Patch location: [x, y] start = ', loc_s, ' end = ', loc_e)

# Major restructuring of code in hopes of making it easier to find errors
@fun.timeit
def run_regression(X, z, reg_string, polydegree, lambdas, N_bs, K, test_size, scale, max_iter=50000):
    """
    Runs the selected regression methods for the input design matrix, p's, lambdas, and using
    the resampling methods as specified.
    While there may be several ways I could have done this more optimally, this function exists
    because a rather late attempt at restructuring the code in order to reduce the amount of duplicate
    lines of code regarding regression, that had just escalated out of control, making it extremely
    difficult to debug and finding whatever was (and maybe is?) causing all the issues.
    :param X:
    :param z:
    :param reg_string:
    :param polydegree:
    :param lambdas:
    :param N_bs:
    :param K:
    :param test_size:
    :param scale:
    :return:
    """
    nlambdas = len(lambdas)
    p = polydegree[-1]

    # Bootstrap and cross-validation
#    N_bootstraps = 100  # int(N/2)
#    K = 5

    # Splitting into train and test, scaling the data
    X_train, X_test, z_train, z_test = fun.split_data(X, z, test_size=test_size)
    X_train_scaled = fun.scale_X(X_train, scale)
    X_test_scaled = fun.scale_X(X_test, scale)
    X_scaled = fun.scale_X(X, scale)

    # Bootstrap arrays
    bs_error_train = np.zeros((p, nlambdas))
    bs_error_test = np.zeros((p, nlambdas))
    bs_bias = np.zeros((p, nlambdas))
    bs_var = np.zeros((p, nlambdas))

    bs_error_train_opt = np.zeros(p)
    bs_error_test_opt = np.zeros(p)
    bs_bias_opt = np.zeros(p)
    bs_var_opt = np.zeros(p)
    bs_lmb_opt = np.zeros(p)

    # Cross-validation arrays
    cv_error_train = np.zeros((p, nlambdas))
    cv_error_test = np.zeros((p, nlambdas))
    cv_error_train_opt = np.zeros(p)
    cv_error_test_opt = np.zeros(p)
    cv_lmb_opt = np.zeros(p)

    # Setting up regression object to be used for regression
    reg_obj = reg.OrdinaryLeastSquares()  # default
    if reg_string == 'SKL':
        # TODO: Remove, 99% sure results with my OLS is accurate
        reg_obj = skl.LinearRegression()  # Testing with SKL OLS
    elif reg_string == 'Ridge':
        reg_obj = reg.RidgeRegression()

    for degree in polydegree:
        n_poly = fun.polynom_N_terms(degree)
        print('p = %2d, np = %3d' % (degree, n_poly))

        # Setting up correct design matrices for the polynomial degree
        X_train_bs = np.zeros((len(X_train_scaled), n_poly))
        X_test_bs = np.zeros((len(X_test_scaled), n_poly))
        X_cv = np.zeros((len(X_scaled), n_poly))

        X_train_bs[:, :] = X_train_scaled[:, 0:n_poly]
        X_test_bs[:, :] = X_test_scaled[:, 0:n_poly]
        X_cv[:, :] = X_scaled[:, 0:n_poly]

        # Looping over all the lambda values
        for i in range(nlambdas):
            lmb = lambdas[i]

            if i % 10 == 0:
                print('i = %d, lmb= %.3e' % (i, lmb))

            if reg_string == 'Ridge':
                reg_obj.set_lambda(lmb)
            elif reg_string == 'Lasso':
                reg_obj = skl.Lasso(alpha=lmb, max_iter=max_iter)

            # Bootstrap
            BS = res.Bootstrap(X_train_bs, X_test_bs, z_train, z_test, reg_obj, fun.mean_squared_error)
            error_, bias_, var_, trainE_ = BS.compute(N_bs)
            bs_error_test[degree-1, i] = error_
            bs_bias[degree-1, i] = bias_
            bs_var[degree-1, i] = var_
            bs_error_train[degree-1, i] = trainE_

            # Cross validation
            CV = res.CrossValidation(X_cv, z, reg_obj, fun.mean_squared_error)
            trainE, testE = CV.compute(K)
            cv_error_train[degree-1, i] = trainE
            cv_error_test[degree-1, i] = testE

        # Locating minimum MSE for each polynomial degree
        # Bootstrap
        index_bs = np.argmin(bs_error_test[degree - 1, :])
        bs_lmb_opt[degree - 1] = lambdas[index_bs]

        # Cross-validation
        index_cv = np.argmin(cv_error_test[degree - 1, :])
        cv_lmb_opt[degree - 1] = lambdas[index_cv]

        print('K-fold cross-validation:')
        print('K=%d, MSE = %.5f' % (K, cv_error_test[degree - 1, i]))

    # Locate minimum MSE  to see how it depends on lambda
    bs_min = np.unravel_index(np.argmin(bs_error_test), bs_error_test.shape)
    cv_min = np.unravel_index(np.argmin(cv_error_test), cv_error_test.shape)
    bs_best = [polydegree[bs_min[0]], lambdas[bs_min[1]]]
    cv_best = [polydegree[cv_min[0]], lambdas[cv_min[1]]]

    # Bootstrap
    bs_error_train_opt[:] = bs_error_train[:, bs_min[1]]
    bs_error_test_opt[:] = bs_error_test[:, bs_min[1]]
    bs_bias_opt[:] = bs_bias[:, bs_min[1]]
    bs_var_opt[:] = bs_var[:, bs_min[1]]

    # Cross-validation
    cv_error_train_opt[:] = cv_error_train[:, cv_min[1]]
    cv_error_test_opt[:] = cv_error_test[:, cv_min[1]]

    # TODO: Combine into 3 or 4 arrays to be returned
    # TODO: currently this is just easier to make sure restructuring actually works
    return (bs_error_train, bs_error_test, bs_bias, bs_var,
            bs_error_train_opt, bs_error_test_opt, bs_bias_opt, bs_var_opt, bs_lmb_opt,
            cv_error_train, cv_error_test, cv_error_train_opt, cv_error_test_opt, cv_lmb_opt,
            bs_min, bs_best, cv_min, cv_best)


########################################################################################################################
if run_mode == 'a':
    # Splitting into train and test data
    X_train, X_test, z_train, z_test = fun.split_data(X, z_ravel, test_size=test_size)  # TODO: check
#    X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=0.2)

    # Scaling the data
    X_train_scaled = fun.scale_X(X_train, scale)
    X_test_scaled = fun.scale_X(X_test, scale)

    # Plotting the Franke function
    fun.plot_surf(x_mesh, y_mesh, z_mesh, 'x', 'y', 'z', 'Franke function, $N$=%d, noise=%.2f' % (N, noise),
                  'franke_n%d' % N, fig_path, run_mode, zlim=[-0.10, 1.40])

    # Ordinary Least Squares
    OLS = reg.OrdinaryLeastSquares()
    betaOLS = OLS.fit(X_train_scaled, z_train)
    z_pred = OLS.predict(X_test_scaled)
    z_fit = OLS.predict(X_train_scaled)

    # Printing MSE and R2 score
    fun.print_MSE_R2(z_test, z_pred, 'test', 'OLS')
    fun.print_MSE_R2(z_train, z_fit, 'train', 'OLS')

    # Confidence interval for beta
    conf = OLS.confidence_interval_beta(X_test, z_test, z_pred)
    fun.plot_confidence_int(betaOLS, conf, 'OLS', fig_path, run_mode)
    plt.show()
    sys.exit(1)


########################################################################################################################
if run_mode == 'b':
    N_samples = N  # number of samples per bootstrap
#    N_bootstraps = 100  # number of resamples
    N_bootstraps = 2
#    N_bootstraps = int(N/2)  # number of resamples
#    N_bootstraps = N

    # Splitting into train and test data
#    print(X[5, :])
    X_train, X_test, z_train, z_test = fun.split_data(X, z_ravel, test_size=test_size)
#    X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=test_size)
#    print('AAA', X_test[196, :])
    print('meow', X_test[190:200, -9:-5])

    # Scaling the data
    X_train_scaled = fun.scale_X(X_train, scale)
    X_test_scaled = fun.scale_X(X_test, scale)
#    print('BBB', X_test_scaled[196, :])
#    X_train_scaled = X_train
#    X_test_scaled = X_test

#    print(X_test_scaled[196, :])

    polydegree = np.arange(1, p + 1)
    trainError = np.zeros(p)
    testError = np.zeros(p)
    bs_error_OLS = np.zeros(p)
    bs_bias_OLS = np.zeros(p)
    bs_var_OLS = np.zeros(p)

    OLS = reg.OrdinaryLeastSquares(OLSmethod)
    for degree in range(1, p + 1):
        n_poly = fun.polynom_N_terms(degree)
        print('p = %2d, np = %3d' % (degree, n_poly))

#        print('CCC', X_test[196, :])

        X_train_bs = np.zeros((len(X_train), n_poly))
        X_test_bs = np.zeros((len(X_test), n_poly))
        X_train_OLS = np.zeros((len(X_train), n_poly))
        X_test_OLS = np.zeros((len(X_test), n_poly))

#        X_train_new[:, :] = X_train_scaled[:, 0:n_poly]
#        X_test_new[:, :] = X_test_scaled[:, 0:n_poly]

        X_train_bs[:, :] = X_train_scaled[:, 0:n_poly]
        X_test_bs[:, :] = X_test_scaled[:, 0:n_poly]

#        print('af hwawa',X_train_bs[195, :])  # why is X_train_bs scaled..............
#        print('DDD', X_test[195, :])

        X_train_OLS[:, :] = X_train_scaled[:, 0:n_poly]
        X_test_OLS[:, :] = X_test_scaled[:, 0:n_poly]

#        print('EEE', X_test[196, :])
#        X_train_OLS = X_train_scaled[:, 0:n_poly]
#        X_test_OLS = X_test_scaled[:, 0:n_poly]

        # Ordinary Least Squares without Bootstrapping
        betaOLS = OLS.fit(X_train_OLS, z_train)
        z_trainOLS = OLS.predict(X_train_OLS)
        z_testOLS = OLS.predict(X_test_OLS)

#        print('FFF', X_test[196, :])
        trainError[degree-1] = fun.mean_squared_error(z_train, z_trainOLS)
        testError[degree-1] = fun.mean_squared_error(z_test, z_testOLS)

        #print('meow', X_test[190:200, -9:-5])
        print('nyaa', X_test_bs[190:200, -9:-5])
        # Bootstrap
#        OLS = reg.OrdinaryLeastSquares(OLSmethod)
        bs = res.Bootstrap(X_train_bs, X_test_bs, z_train, z_test, OLS, fun.mean_squared_error)
        error_OLS, bias_OLS, var_OLS = bs.compute(N_bootstraps, test=True)
        bs_error_OLS[degree-1] = error_OLS
        bs_bias_OLS[degree-1] = bias_OLS
        bs_var_OLS[degree-1] = var_OLS
#        print(bias_OLS, type(bias_OLS))
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
    '''
    fun.plot_MSE_train_test(polydegree, trainError, bs_error_OLS, n_franke, test_size, noise,
                            fig_path, run_mode, 'Bootstrap')

    fun.plot_MSE_train_test(polydegree, error_train_BS, error_test_BS,
                            '%s, N = %d, $N_{BS}$ = %d' % (reg_g, N, N_bootstraps),
                            '%s_patch%d_n%d_ts%.2f' % (reg_g, patch, N, test_size),
                            fig_path, run_mode, resample='Bootstrap')

    fun.plot_MSE_train_test(polydegree, trainError, testError, n_franke, test_size, noise,
                            fig_path, run_mode)

#    fun.plot_MSE_test_OLS_fit(polydegree, trainError, testError, n_franke, test_size, noise, OLSmethod)
    '''
    fun.plot_bias_variance(polydegree, bs_error_OLS, bs_bias_OLS, bs_var_OLS,
                           'trade-off, $N$=%d, $N_{bs}$=%d' % (N, N_bootstraps),
                           'N%d_Nbs%d' % (N, N_bootstraps), fig_path, run_mode)

    plt.show()


########################################################################################################################
if run_mode == 'c':
    K = 5

    polydegree = np.arange(1, p + 1)
    error_train_CV = np.zeros(len(polydegree))
    error_test_CV = np.zeros(len(polydegree))
    error_train_SKL = np.zeros(len(polydegree))
    error_test_SKL = np.zeros(len(polydegree))

    X_scaled = fun.scale_X(X, scale)
    OLS = reg.OrdinaryLeastSquares(OLSmethod)
    for degree in range(1, p + 1):
        n_poly = fun.polynom_N_terms(degree)
        print('p = %2d, np = %3d' % (degree, n_poly))

        X_cv = np.zeros((len(X_scaled), n_poly))
        X_cv[:, :] = X_scaled[:, 0:n_poly]

        CV = res.CrossValidation(X_cv, z_ravel, OLS, fun.mean_squared_error)
        trainE, testE = CV.compute(K)
        error_train_CV[degree-1] = trainE
        error_test_CV[degree-1] = testE

        print('K-fold cross-validation:')
        print('K=%d, MSE = %.5f' % (K, error_test_CV[degree-1]))

        CV_SKL = res.CrossValidationSKL(X_cv, z_ravel, OLS)
        trainE, testE = CV_SKL.compute(K)
        error_train_SKL[degree-1] = trainE
        error_test_SKL[degree-1] = testE

        print('K-fold cross-validation SKL:')
        print('K=%d, MSE = %.5f' % (K, error_test_SKL[degree-1]))

    fun.plot_multiple_y(polydegree, [error_test_CV, error_test_SKL], ['test CV', 'test SKL'],
                        'Comparing different CV implementations', 'Polynomial degree', 'Mean squared error',
                        'test_compare_CV_SKL_p%d' % p, fig_path, run_mode)

    fun.plot_multiple_y(polydegree, [error_train_CV, error_train_SKL], ['train CV', 'train SKL'],
                        'Comparing different CV implementations', 'Polynomial degree', 'Mean squared error',
                        'train_compare_CV_SKL_p%d' % p, fig_path, run_mode)

    # Suspect the difference lies in the splitting, which would be worth looking at

    plt.show()


########################################################################################################################
if run_mode == 'd':
    # Setting up for Ridge regression
    nlambdas = 20  # 100
    lambdas = np.logspace(-4, 1, nlambdas)

    # Bootstrap
    N_bootstraps = 100  # int(N/2)  # 100

    # Cross-validation
    K = 5

    polydegree = np.arange(1, p + 1)

    variables = run_regression(X, z_ravel, 'Ridge', polydegree, lambdas, N_bootstraps, K, test_size, scale)
    # Unpacking variables
    bs_error_train, bs_error_test = variables[0:2]
    bs_bias, bs_var = variables[2:4]
    bs_error_train_opt, bs_error_test_opt = variables[4:6]
    bs_bias_opt, bs_var_opt, bs_lmb_opt = variables[6:9]
    cv_error_train, cv_error_test = variables[9:11]
    cv_error_train_opt, cv_error_test_opt, cv_lmb_opt = variables[11:14]
    bs_min, bs_best, cv_min, cv_best = variables[14:18]

    # Plotting

    fun.plot_lambda_mse(lambdas, bs_error_test[bs_min[0], :], 'Bootstrap p=%d' % bs_best[0],
                        'bootstrap_p%d' % bs_best[0], fig_path, run_mode, fs=14)

    fun.plot_lambda_mse(lambdas, cv_error_test[cv_min[0], :], 'Cross-validation p=%d' % cv_best[0],
                        'cv_p%d' % cv_best[0], fig_path, run_mode, fs=14)

    # Bootstrap Plots
#    fun.plot_bias_variance(polydegree, bs_error_test[:, bs_min_error[1]], bs_bias[:, bs_min_error[1]], bs_var[:, bs_min_error[1]],
    fun.plot_bias_variance(polydegree, bs_error_test_opt, bs_bias_opt, bs_var_opt,
                           'Ridge regression, $\lambda$=%.2e, $N$=%d, $N_{bs}$=%d' % (bs_best[1], N, N_bootstraps),
                           'N%d_Nbs%d' % (N, N_bootstraps), fig_path, run_mode)

    fun.plot_degree_lambda(polydegree, bs_lmb_opt, 'Minimum MSE, Bootstrap',
                           'bootstrap', fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, bs_error_test, 'MSE', 'Bootstrap + Ridge',
                     'bs_ridge_error', fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, bs_bias, 'bias', 'Bootstrap + Ridge',
                     'bs_ridge_bias', fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, bs_var, 'var', 'Bootstrap + Ridge',
                     'bs_ridge_variance', fig_path, run_mode)

    # Cross-Validation Plots
    fun.plot_degree_lambda(polydegree, cv_lmb_opt, 'Minimum MSE, CV',
                           'cv', fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, cv_error_test, 'MSE', 'CV + Ridge',
                     'cv_ridge_error', fig_path, run_mode)

    # MSE train test plots
    fun.plot_MSE_train_test(polydegree, cv_error_train_opt, cv_error_test_opt,
                            'p=%d, $\lambda$=%.2e, N=%d, $N_{BS}$=%d' % (cv_best[0], cv_best[1], N, N_bootstraps),
                            'p%d_lmb%.2e_n%d_ts%.2f' % (cv_best[0], cv_best[1], N, test_size),
                            fig_path, run_mode, resample='CV')

    fun.plot_MSE_train_test(polydegree, bs_error_train_opt, bs_error_test_opt,
                            'p=%d, $\lambda$=%.2e, N=%d, $N_{BS}$=%d' % (bs_best[0], bs_best[1], N, N_bootstraps),
                            'p%d_lmb%.2e_n%d_ts%.2f' % (bs_best[0], bs_best[1], N, test_size),
                            fig_path, run_mode, resample='Bootstrap')

    fun.plot_multiple_y(polydegree, [bs_error_test_opt, cv_error_test_opt], ['Bootstrap', 'CV'],
                        'Comparing bootstrap and cross-validation', 'Polynomial degree', 'Mean squared error',
                        'test_compare_BS_CV_p%d' % p, fig_path, run_mode)

    # Confidence Interval plot for Ridge
    X_train, X_test, z_train, z_test = fun.split_data(X, z_ravel, test_size=test_size)
    X_train_scaled = fun.scale_X(X_train, scale)
    X_test_scaled = fun.scale_X(X_test, scale)

    Ridge = reg.RidgeRegression()
    Ridge.set_lambda(1e-4)
    n_poly = fun.polynom_N_terms(6)
    Ridge.fit(X_train_scaled[:, 0:n_poly], z_train)
    z_Ridge = Ridge.predict(X_test_scaled[:, 0:n_poly])
    beta_Ridge = Ridge.beta
    conf = Ridge.confidence_interval_beta(X_test_scaled[:, 0:n_poly], z_test, z_Ridge)
    fun.plot_confidence_int(beta_Ridge, conf, 'Ridge', fig_path, run_mode)
    plt.show()


########################################################################################################################
if run_mode == 'e':
    # Lambdas / alphas for LASSO regression
    nlambdas = 15#30  # 100
    lambdas = np.logspace(-4, -1, nlambdas)

    # Bootstrap and cross-validation
    N_bootstraps = int(N/2)
    K = 5

    polydegree = np.arange(1, p + 1)

    variables = run_regression(X, z_ravel, 'Lasso', polydegree, lambdas, N_bootstraps, K, test_size, scale, max_iter=50000)
    # Unpacking variables
    bs_error_train, bs_error_test = variables[0:2]
    bs_bias, bs_var = variables[2:4]
    bs_error_train_opt, bs_error_test_opt = variables[4:6]
    bs_bias_opt, bs_var_opt, bs_lmb_opt = variables[6:9]
    cv_error_train, cv_error_test = variables[9:11]
    cv_error_train_opt, cv_error_test_opt, cv_lmb_opt = variables[11:14]
    bs_min, bs_best, cv_min, cv_best = variables[14:18]
    # way too many

    # Plotting

    fun.plot_lambda_mse(lambdas, bs_error_test[bs_min[0], :], 'Bootstrap p=%d' % bs_best[0],
                        'bootstrap_p%d' % bs_best[0], fig_path, run_mode, fs=14)

    fun.plot_lambda_mse(lambdas, cv_error_test[cv_min[0], :], 'Cross-validation p=%d' % cv_best[0],
                        'cv_p%d' % cv_best[0], fig_path, run_mode, fs=14)

    # Bootstrap Plots
    #    fun.plot_bias_variance(polydegree, bs_error_test[:, bs_min_error[1]], bs_bias[:, bs_min_error[1]], bs_var[:, bs_min_error[1]],
    fun.plot_bias_variance(polydegree, bs_error_test_opt, bs_bias_opt, bs_var_opt,
                           'Lasso regression, $\lambda$=%.2e, $N$=%d, $N_{bs}$=%d' % (bs_best[1], N, N_bootstraps),
                           'N%d_Nbs%d' % (N, N_bootstraps), fig_path, run_mode)

    fun.plot_degree_lambda(polydegree, bs_lmb_opt, 'Minimum MSE, Bootstrap',
                           'bootstrap', fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, bs_error_test, 'MSE', 'Bootstrap + Lasso',
                     'bs_lasso_error', fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, bs_bias, 'bias', 'Bootstrap + Lasso',
                     'bs_lasso_bias', fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, bs_var, 'var', 'Bootstrap + Lasso',
                     'bs_lasso_variance', fig_path, run_mode)

    # Cross-Validation Plots
    fun.plot_degree_lambda(polydegree, cv_lmb_opt, 'Minimum MSE, CV',
                           'cv', fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, cv_error_test, 'MSE', 'CV + Lasso',
                     'cv_lasso_error', fig_path, run_mode)

    # MSE train test plots
    fun.plot_MSE_train_test(polydegree, cv_error_train_opt, cv_error_test_opt,
                            'p=%d, $\lambda$=%.2e, N=%d, $N_{BS}$=%d' % (cv_best[0], cv_best[1], N, N_bootstraps),
                            'p%d_lmb%.2e_n%d_ts%.2f' % (cv_best[0], cv_best[1], N, test_size),
                            fig_path, run_mode, resample='CV')

    fun.plot_MSE_train_test(polydegree, bs_error_train_opt, bs_error_test_opt,
                            'p=%d, $\lambda$=%.2e, N=%d, $N_{BS}$=%d' % (bs_best[0], bs_best[1], N, N_bootstraps),
                            'p%d_lmb%.2e_n%d_ts%.2f' % (bs_best[0], bs_best[1], N, test_size),
                            fig_path, run_mode, resample='Bootstrap')

    fun.plot_heatmap(lambdas, polydegree, bs_error_test, 'MSE', 'Bootstrap + Lasso',
                     'bs_lasso_error', fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, cv_error_test, 'MSE', 'Cross-validation + Lasso',
                     'cv_lasso_error', fig_path, run_mode)

    plt.show()


########################################################################################################################
if run_mode == 'f':
    # fetch terrain data and plot it i guess
    fun.plot_terrain(z_mesh, 'Terrain, location [%d, %d] to [%d, %d]' % (loc_s[0], loc_s[1], loc_e[0], loc_e[1]),
                     'terrain_n%d_%d_%d' % (n_terrain, loc_s[0], loc_s[1]), fig_path, run_mode)

    fun.plot_surf(y_mesh, x_mesh, z_mesh, 'x', 'y', 'z', 'Terrain, $N$=%d' % N,
                  'surf_terrain_n%d_%d_%d' % (n_terrain, loc_s[0], loc_s[1]), fig_path, run_mode, [0, 1.0])

    plt.show()


########################################################################################################################
if run_mode == 'g':

    if reg_g == 'OLS':
        # Setting to 1 for OLS
        nlambdas = 1
        lambdas = np.zeros(1)
    elif reg_g == 'Ridge':
        # Lambdas for Ridge regression
        nlambdas = 15  # 30  # 100
        lambdas = np.logspace(-4, 1, nlambdas)
    elif reg_g == 'Lasso':
        # Lambdas / alphas for Lasso regression
        nlambdas = 15  # 30  # 100
        lambdas = np.logspace(-4, 1, nlambdas)
        max_iter = 100000

    polydegree = np.arange(1, p + 1)
    N_bootstraps = 100
    K = 5

    variables = run_regression(X, z, reg_g, polydegree, lambdas, N_bootstraps, K, test_size, scale, max_iter=max_iter)
    # Unpacking variables
    bs_error_train, bs_error_test = variables[0:2]
    bs_bias, bs_var = variables[2:4]
    bs_error_train_opt, bs_error_test_opt = variables[4:6]
    bs_bias_opt, bs_var_opt, bs_lmb_opt = variables[6:9]
    cv_error_train, cv_error_test = variables[9:11]
    cv_error_train_opt, cv_error_test_opt, cv_lmb_opt = variables[11:14]
    bs_min, bs_best, cv_min, cv_best = variables[14:18]
    # way too many

    # Plotting
    save = '%s_patch%d_N%d_Nbs%d_pmax%d_nlambdas%d' % (reg_g, patch, n_terrain, N_bootstraps, p, nlambdas)

    fun.plot_lambda_mse(lambdas, bs_error_test[bs_min[0], :], 'Bootstrap p=%d' % bs_best[0],
                        'bootstrap_p%d_%s' % (bs_best[0], save), fig_path, run_mode, fs=14)

    fun.plot_lambda_mse(lambdas, cv_error_test[cv_min[0], :], 'Cross-validation p=%d' % cv_best[0],
                        'cv_p%d_%s' % (cv_best[0], save), fig_path, run_mode, fs=14)

    # Bootstrap Plots
    #    fun.plot_bias_variance(polydegree, bs_error_test[:, bs_min_error[1]], bs_bias[:, bs_min_error[1]], bs_var[:, bs_min_error[1]],
    fun.plot_bias_variance(polydegree, bs_error_test_opt, bs_bias_opt, bs_var_opt,
                           '%s, $\lambda$=%.2e, $N$=%d, $N_{bs}$=%d' % (reg_g, bs_best[1], N, N_bootstraps),
                           'lmb_%.2e_%s' % (bs_best[1], save), fig_path, run_mode)

    fun.plot_degree_lambda(polydegree, bs_lmb_opt, 'Minimum MSE, Bootstrap',
                           'bootstrap_p%.2e_%s' % (bs_best[1], save), fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, bs_error_test, 'MSE', 'Bootstrap + %s' % reg_g,
                     'bs_error_%s' % save, fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, bs_bias, 'bias', 'Bootstrap + %s' % reg_g,
                     'bs_bias_%s' % save, fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, bs_var, 'var', 'Bootstrap + %s' % reg_g,
                     'bs_variance_%s' % save, fig_path, run_mode)

    # Cross-Validation Plots
    fun.plot_degree_lambda(polydegree, cv_lmb_opt, 'Minimum MSE, CV',
                           'cv_lmb%.2e_%s' % (cv_best[1], save), fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, cv_error_test, 'MSE', 'CV + %s' % reg_g,
                     'cv_error_%s' % save, fig_path, run_mode)

    # MSE train test plots
    fun.plot_MSE_train_test(polydegree, cv_error_train_opt, cv_error_test_opt,
                            'p=%d, $\lambda$=%.2e, N=%d, $N_{BS}$=%d' % (cv_best[0], cv_best[1], N, N_bootstraps),
                            'p%d_lmb%.2e_n%d_ts%.2f' % (cv_best[0], cv_best[1], N, test_size),
                            fig_path, run_mode, resample='CV')

    fun.plot_MSE_train_test(polydegree, bs_error_train_opt, bs_error_test_opt,
                            'p=%d, $\lambda$=%.2e, N=%d, $N_{BS}$=%d' % (bs_best[0], bs_best[1], N, N_bootstraps),
                            'p%d_lmb%.2e_n%d_ts%.2f' % (bs_best[0], bs_best[1], N, test_size),
                            fig_path, run_mode, resample='Bootstrap')

    plt.show()