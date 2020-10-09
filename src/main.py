import functions as fun
import regression_methods as reg
import resampling_methods as res
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
import sys
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


# Input used instead of sys.argv, as changing arguments with PyCharm takes some time
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
# Most of the parameters that can be adjusted will be inside the if-test corresponding to the data or task
p_dict = {'a': 6, 'b': 20, 'c': 20, 'd': 20,
          'e': 20, 'f': 1, 'g': 20}
scale_dict = {'a': [True, False], 'b': [True, False], 'c': [True, False], 'd': [True, False],
              'e': [True, False], 'f': None, 'g': [True, False]}
p = p_dict[run_mode]  # degree of polynomial for the task
scale = scale_dict[run_mode]  # first index is whether to subtract mean, second is to scale by std

test_size = 0.2
fig_path = '../figures/'
data_path = '../datafiles/'
write_path = '../datafiles/'

# For the terrain data / g, every other task has their own reg_str defined
reg_str = 'OLS'
#reg_str = 'Ridge'
#reg_str = 'Lasso'

# Benchmark settings
benchmark = False  # setting to True will adjust all relevant settings for all task
if benchmark is True:
    p = 5
    scale = [True, False]
    reg_str = 'OLS'

########################################################################################################################
if data == 'franke':
    # Creating data set for the Franke function tasks
    seed = 4155
    np.random.seed(seed)
    n_franke = 23  # 529 points
    N = n_franke**2  # Total number of samples n*2
    noise = 0.05  # noise level

    if benchmark is True:
        n_franke = 23
        N = 529
        noise = 0.05

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


########################################################################################################################
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


########################################################################################################################
#@fun.timeit
@ignore_warnings(category=ConvergenceWarning)
def run_regression(X, z, reg_string, polydegree, lambdas, N_bs, K, test_size, scale, max_iter=50000):
    """
    Runs the selected regression methods for the input design matrix, p's, lambdas, and using
    the resampling methods as specified.
    While there may be several ways I could have done this more optimally, this function exists
    because a rather late attempt at restructuring the code in order to reduce the amount of duplicate
    lines of code regarding regression, that had just escalated out of control, making it extremely
    difficult to debug and finding whatever was causing all the issues.
    Turns out the real error was all the friends we made along the way.
    :param X: (N, p) array containing input design matrix
    :param z: (N, 1) array containing data points
    :param reg_string: string containing the name of the regression method to be used
    :param polydegree: list/range of the different p-values to be used
    :param lambdas: array of all the lambda values to be used
    :param N_bs: int, number of Bootstraps
    :param K: int, number of folds in the Cross-Validation
    :param test_size: float, size of the test partition [0.0, 1.0]
    :param scale: list determining if the scaling is only by the mean, the std or both [bool(mean), bool(std)]
    :param max_iter: maximum number of iterations for Lasso
    :return: a lot of arrays with the various results and different ways of representing the data
    """
    nlambdas = len(lambdas)  # number of lambdas
    p = polydegree[-1]  # the maximum p-value
    method = 4  # OLS method

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

    bs_error_train_opt = np.zeros((p, 2))
    bs_error_test_opt = np.zeros((p, 2))
    bs_bias_opt = np.zeros((p, 2))  # First index is min(MSE) lmb for each p, second at lmb that yields total lowest MSE
    bs_var_opt = np.zeros((p, 2))
    bs_lmb_opt = np.zeros(p)

    # Cross-validation arrays
    cv_error_train = np.zeros((p, nlambdas))
    cv_error_test = np.zeros((p, nlambdas))
    cv_error_train_opt = np.zeros((p, 2))
    cv_error_test_opt = np.zeros((p, 2))
    cv_lmb_opt = np.zeros(p)

    # Setting up regression object to be used for regression (Lasso is dealt with later)
    reg_obj = reg.OrdinaryLeastSquares(method)  # default
    if reg_string == 'SKL':
        reg_obj = skl.LinearRegression()  # Testing with scikit-learn OLS
    elif reg_string == 'Ridge':
        reg_obj = reg.RidgeRegression()

    # Looping over all polynomial degrees in the analysis
    for degree in polydegree:
        n_poly = fun.polynom_N_terms(degree)  # number of terms in the design matrix for the given degree
        print('p = %2d, np = %3d' % (degree, n_poly))

        # Setting up correct design matrices for the current degree
        X_train_bs = np.zeros((len(X_train_scaled), n_poly))
        X_test_bs = np.zeros((len(X_test_scaled), n_poly))
        X_cv = np.zeros((len(X_scaled), n_poly))

        # Filling the elements up to term n_poly
        X_train_bs[:, :] = X_train_scaled[:, 0:n_poly]
        X_test_bs[:, :] = X_test_scaled[:, 0:n_poly]
        X_cv[:, :] = X_scaled[:, 0:n_poly]

        # Looping over all the lambda values
        for i in range(nlambdas):
            lmb = lambdas[i]  # current lambda value

            # Printing out in order to gauge where we are
            if i % 10 == 0:
                print('i = %d, lmb= %.3e' % (i, lmb))

            # Updating the current lambda value for Ridge and Lasso
            if reg_string == 'Ridge':
                reg_obj.set_lambda(lmb)
            elif reg_string == 'Lasso':
                reg_obj = skl.Lasso(alpha=lmb, max_iter=max_iter, precompute=True, warm_start=True)

            # Bootstrap
            BS = res.Bootstrap(X_train_bs, X_test_bs, z_train, z_test, reg_obj)
            error_, bias_, var_, trainE_ = BS.compute(N_bs)  # performing the Bootstrap
            bs_error_test[degree-1, i] = error_
            bs_bias[degree-1, i] = bias_
            bs_var[degree-1, i] = var_
            bs_error_train[degree-1, i] = trainE_

            # Cross validation
            CV = res.CrossValidation(X_cv, z, reg_obj)
            trainE, testE = CV.compute(K)  # performing the Cross-Validation
            cv_error_train[degree-1, i] = trainE
            cv_error_test[degree-1, i] = testE

        # Locating minimum MSE for each polynomial degree
        # Bootstrap
        index_bs = np.argmin(bs_error_test[degree - 1, :])
        bs_lmb_opt[degree - 1] = lambdas[index_bs]
        bs_error_train_opt[:, 0] = bs_error_train[:, index_bs]
        bs_error_test_opt[:, 0] = bs_error_test[:, index_bs]
        bs_bias_opt[:, 0] = bs_bias[:, index_bs]
        bs_var_opt[:, 0] = bs_var[:, index_bs]

        # Cross-validation
        index_cv = np.argmin(cv_error_test[degree - 1, :])
        cv_lmb_opt[degree - 1] = lambdas[index_cv]
        cv_error_train_opt[:, 0] = cv_error_train[:, index_cv]
        cv_error_test_opt[:, 0] = cv_error_test[:, index_cv]

    # Locate minimum MSE  to see how it depends on lambda
    bs_min = np.unravel_index(np.argmin(bs_error_test), bs_error_test.shape)
    cv_min = np.unravel_index(np.argmin(cv_error_test), cv_error_test.shape)
    bs_best = [polydegree[bs_min[0]], lambdas[bs_min[1]]]
    cv_best = [polydegree[cv_min[0]], lambdas[cv_min[1]]]

    # Bootstrap
    bs_error_train_opt[:, 1] = bs_error_train[:, bs_min[1]]
    bs_error_test_opt[:, 1] = bs_error_test[:, bs_min[1]]
    bs_bias_opt[:, 1] = bs_bias[:, bs_min[1]]
    bs_var_opt[:, 1] = bs_var[:, bs_min[1]]

    # Cross-validation
    cv_error_train_opt[:, 1] = cv_error_train[:, cv_min[1]]
    cv_error_test_opt[:, 1] = cv_error_test[:, cv_min[1]]

    # This return is extremely large, sadly, and should have been improved upon
    # this was just the fastest way of doing it when I had to restructure the code
    # so better planning in the future would be a better solution
    return (bs_error_train, bs_error_test, bs_bias, bs_var,
            bs_error_train_opt, bs_error_test_opt, bs_bias_opt, bs_var_opt, bs_lmb_opt,
            cv_error_train, cv_error_test, cv_error_train_opt, cv_error_test_opt, cv_lmb_opt,
            bs_min, bs_best, cv_min, cv_best)


########################################################################################################################
if run_mode == 'a':
    save = 'N%d_nf%d_p%d_noise%.2f_seed%d' % (N, n_franke, p, noise, seed)

    # Splitting into train and test data
    X_train, X_test, z_train, z_test = fun.split_data(X, z_ravel, test_size=test_size)
#    X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=test_size)

    # Scaling the data
    X_train_scaled = fun.scale_X(X_train, scale)
    X_test_scaled = fun.scale_X(X_test, scale)

    # Plotting the Franke function
    fun.plot_surf(x_mesh, y_mesh, z_mesh, 'x', 'y', 'z', 'Franke function, $N$=%d, noise=%.2f' % (N, noise),
                  'franke_%s' % save, fig_path, run_mode, zlim=[-0.10, 1.40], azim=60)

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
    fun.plot_confidence_int(betaOLS, conf, 'OLS', save, fig_path, run_mode)
    plt.show()


########################################################################################################################
if run_mode == 'b':
    # Originally this only performed bootstrap with OLS
    # But now makes every relevant Franke function plot for OLS
    reg_str = 'OLS'  # alternatively SKL for scikit-learns OLS

    # Bootstrap variables
    N_bootstraps = int(N/2)  # number of resamples (ex. N/2, N/4)

    # Cross-validation
    K = 5

    # Benchmark settings
    if benchmark is True:
        N_bootstraps = 264
        K = 5

    # Setting up to make sure things work
    nlambdas = 1
    lambdas = np.ones(nlambdas)

    # Parameters for saving to file
    save = 'N%d_pmax%d_nlamb%d_noise%.2f_seed%d' % (N, p, nlambdas, noise, seed)
    save_bs = '%s_%s_%s_Nbs%d' % (save, reg_str, 'boot', N_bootstraps)
    save_cv = '%s_%s_%s_k%d' % (save, reg_str, 'cv', K)

    # Performing the regression
    polydegree = np.arange(1, p + 1)
    variables = run_regression(X, z_ravel, reg_str, polydegree, lambdas, N_bootstraps, K, test_size, scale)
    # Unpacking variables
    bs_error_train, bs_error_test = variables[0:2]
    bs_bias, bs_var = variables[2:4]
    bs_error_train_opt, bs_error_test_opt = variables[4:6]
    bs_bias_opt, bs_var_opt, bs_lmb_opt = variables[6:9]
    cv_error_train, cv_error_test = variables[9:11]
    cv_error_train_opt, cv_error_test_opt, cv_lmb_opt = variables[11:14]
    bs_min, bs_best, cv_min, cv_best = variables[14:18]

    # Bootstrap plots
    xlim = [1, 20]
    ylim = [0.0, 0.02]
    fun.plot_MSE_train_test(polydegree, bs_error_train_opt[:, 0], bs_error_test_opt[:, 0],
                            '%s, $N$=%d, $N_{bs}$=%d, noise=%.2f' % (reg_str, N, N_bootstraps, noise),
                            'train_test_%s' % save_bs, fig_path, run_mode,
                            resample='Bootstrap', xlim=xlim, ylim=ylim)

    fun.plot_bias_variance(polydegree, bs_error_test_opt[:, 0], bs_bias_opt[:, 0], bs_var_opt[:, 0],
                           'Bootstrap, %s, $N$=%d, $N_{bs}$=%d, noise=%.2f' % (reg_str, N, N_bootstraps, noise),
                           '%s' % save_bs, fig_path, run_mode, xlim=xlim, ylim=ylim)

    # Cross-validation plot
    fun.plot_MSE_train_test(polydegree, cv_error_train_opt[:, 0], cv_error_test_opt[:, 0],
                            '%s, $N$=%d, $K$=%d, noise=%.2f' % (reg_str, N, K, noise),
                            'train_test_%s' % save_cv, fig_path, run_mode,
                            resample='CV', xlim=xlim, ylim=ylim)

    # Write bootstrap to file
    fun.save_to_file([bs_error_test_opt[:, 0], bs_bias_opt[:, 0], bs_var_opt[:, 0]],
                     ['bs_error_test', 'bs_bias', 'bs_var'],
                     write_path+'franke/bias_var_task_%s_%s.txt' % (run_mode, save_bs), benchmark)

    # Write CV to file
    fun.save_to_file([cv_error_test_opt[:, 0], cv_error_train[:, 0]], ['cv_error_test', 'cv_error_train'],
                     write_path+'franke/train_test_task_%s_%s.txt' % (run_mode, save_cv), benchmark)

    plt.show()


########################################################################################################################
if run_mode == 'c':
    # Performs Cross-Validation with OLS
    K = 5

    polydegree = np.arange(1, p + 1)
    error_train_CV = np.zeros(len(polydegree))
    error_test_CV = np.zeros(len(polydegree))
    error_train_SKL = np.zeros(len(polydegree))
    error_test_SKL = np.zeros(len(polydegree))

    X_scaled = fun.scale_X(X, scale)
    OLS = reg.OrdinaryLeastSquares(method)
    for degree in range(1, p + 1):
        n_poly = fun.polynom_N_terms(degree)
        print('p = %2d, np = %3d' % (degree, n_poly))

        X_cv = np.zeros((len(X_scaled), n_poly))
        X_cv[:, :] = X_scaled[:, 0:n_poly]

        CV = res.CrossValidation(X_cv, z_ravel, OLS)
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

    plt.show()


########################################################################################################################
if run_mode == 'd':
    # Performs Lasso regression for a set of p and lambdas, with Bootstrap and CV
    reg_str = 'Ridge'
    nlambdas = 20
    lambdas = np.logspace(-6, 1, nlambdas)

    # Bootstrap
    N_bootstraps = int(N/2)

    # Cross-validation
    K = 5

    # Benchmark settings
    if benchmark is True:
        N_bootstraps = 264
        K = 5
        nlambdas = 10
        lambdas = np.logspace(-4, 1, nlambdas)

    # Parameters for easier saving to file
    save = 'N%d_pmax%d_nlamb%d_noise%.2f_seed%d' % (N, p, nlambdas, noise, seed)
    save_bs = '%s_%s_%s_Nbs%d' % (save, reg_str, 'boot', N_bootstraps)
    save_cv = '%s_%s_%s_k%d' % (save, reg_str, 'cv', K)
    save_bc = '%s_%s_Nbs%d_k%d' % (save, reg_str, N_bootstraps, K)

    polydegree = np.arange(1, p + 1)

    variables = run_regression(X, z_ravel, reg_str, polydegree, lambdas, N_bootstraps, K, test_size, scale)
    # Unpacking variables
    bs_error_train, bs_error_test = variables[0:2]
    bs_bias, bs_var = variables[2:4]
    bs_error_train_opt, bs_error_test_opt = variables[4:6]
    bs_bias_opt, bs_var_opt, bs_lmb_opt = variables[6:9]
    cv_error_train, cv_error_test = variables[9:11]
    cv_error_train_opt, cv_error_test_opt, cv_lmb_opt = variables[11:14]
    bs_min, bs_best, cv_min, cv_best = variables[14:18]

    # Plotting

    # Lambda, MSE plots
    fun.plot_lambda_mse(lambdas, bs_error_test[bs_min[0], :], '%s, Bootstrap, p=%d' % (reg_str, bs_best[0]),
                        'bootstrap_p%d_%s' % (bs_best[0], save_bs), fig_path, run_mode, fs=14)

    fun.plot_lambda_mse(lambdas, cv_error_test[cv_min[0], :], '%s, Cross-validation, p=%d' % (reg_str, cv_best[0]),
                        'cv_p%d_%s' % (cv_best[0], save_cv), fig_path, run_mode, fs=14)

    # Polydegree, lambda plots
    fun.plot_degree_lambda(polydegree, bs_lmb_opt, 'Minimum MSE, Bootstrap',
                           save_bs, fig_path, run_mode)

    fun.plot_degree_lambda(polydegree, cv_lmb_opt, 'Minimum MSE, CV',
                           save_cv, fig_path, run_mode)

    # Heatmaps
    fun.plot_heatmap(lambdas, polydegree, bs_error_test, 'MSE', 'Bootstrap + %s' % reg_str,
                     'bs_error_%s' % save_bs, fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, bs_bias, 'bias', 'Bootstrap + %s' % reg_str,
                     'bs_bias_%s' % save_bs, fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, bs_var, 'var', 'Bootstrap + %s' % reg_str,
                     'bs_variance_%s' % save_bs, fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, cv_error_test, 'MSE', 'CV + %s' % reg_str,
                     'cv_error_%s' % save_cv, fig_path, run_mode)

    # Bias-variance trade-off
    i = 1
    fun.plot_bias_variance(polydegree, bs_error_test_opt[:, i], bs_bias_opt[:, i], bs_var_opt[:, i],
                           '%s, $\lambda$=%.2e, $N$=%d, $N_{bs}$=%d' % (reg_str, bs_best[1], N, N_bootstraps),
                           '%s_opt%d' % (save_bs, i), fig_path, run_mode)

    # MSE train test plots
    fun.plot_MSE_train_test(polydegree, bs_error_train_opt[:, i], bs_error_test_opt[:, i],
                            'p=%d, $\lambda$=%.2e, N=%d, $N_{BS}$=%d' % (bs_best[0], bs_best[1], N, N_bootstraps),
                            '%s_opt%d' % (save_bs, i),
                            fig_path, run_mode, resample='Bootstrap')

    fun.plot_MSE_train_test(polydegree, cv_error_train_opt[:, i], cv_error_test_opt[:, i],
                            'p=%d, $\lambda$=%.2e, N=%d, $N_{BS}$=%d' % (cv_best[0], cv_best[1], N, N_bootstraps),
                            '%s_opt%d' % (save_cv, i),
                            fig_path, run_mode, resample='CV')

    fun.plot_multiple_y(polydegree, [bs_error_test_opt[:, i], cv_error_test_opt[:, i]],
                        ['test, Bootstrap', 'test, CV'],
                        'Comparing bootstrap and cross-validation', 'Polynomial degree', 'Mean squared error',
                        'test_compare_BS_CV_p%d_%s_opt%d' % (p, save_bc, i), fig_path, run_mode)

    # Write bootstrap to file
    fun.save_to_file([bs_error_test_opt[:, 0], bs_bias_opt[:, 0], bs_var_opt[:, 0], bs_lmb_opt],
                     ['bs_error_test', 'bs_bias', 'bs_var', 'bs_lmb'],
                     write_path+'franke/bias_var_task_%s_%s.txt' % (run_mode, save_bs), benchmark)

    # Write CV to file
    fun.save_to_file([cv_error_test_opt[:, 0], cv_error_train[:, 0], cv_lmb_opt],
                     ['cv_error_test', 'cv_error_train', 'cv_lmb'],
                     write_path+'franke/train_test_task_%s_%s.txt' % (run_mode, save_cv), benchmark)

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
    fun.plot_confidence_int(beta_Ridge, conf, 'Ridge', save, fig_path, run_mode)
    plt.show()


########################################################################################################################
if run_mode == 'e':
    # Performs Lasso regression for a set of p and lambdas, with Bootstrap and CV
    reg_str = 'Lasso'
    max_iter = 100000

    # Lambdas / alphas for LASSO regression
    nlambdas = 10
    lambdas = np.logspace(-6, -1, nlambdas)

    # Bootstrap and cross-validation
    N_bootstraps = int(N/2)
    K = 5

    # Benchmark settings
    if benchmark is True:
        N_bootstraps = int(N / 2)
        K = 5
        max_iter = 50000
        nlambdas = 5
        lambdas = np.logspace(-4, 0, nlambdas)

    # Parameters for easier saving to file
    save = 'N%d_pmax%d_nlamb%d_noise%.2f_seed%d' % (N, p, nlambdas, noise, seed)
    save_bs = '%s_%s_%s_Nbs%d' % (save, reg_str, 'boot', N_bootstraps)
    save_cv = '%s_%s_%s_k%d' % (save, reg_str, 'cv', K)
    save_bc = '%s_%s_Nbs%d_k%d' % (save, reg_str, N_bootstraps, K)

    polydegree = np.arange(1, p + 1)

    variables = run_regression(X, z_ravel, reg_str, polydegree, lambdas, N_bootstraps, K,
                               test_size, scale, max_iter=max_iter)
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

    # Lambda, MSE plots
    fun.plot_lambda_mse(lambdas, bs_error_test[bs_min[0], :], '%s, Bootstrap, p=%d' % (reg_str, bs_best[0]),
                        'bootstrap_p%d_%s' % (bs_best[0], save_bs), fig_path, run_mode, fs=14)

    fun.plot_lambda_mse(lambdas, cv_error_test[cv_min[0], :], '%s, Cross-validation, p=%d' % (reg_str, cv_best[0]),
                        'cv_p%d_%s' % (cv_best[0], save_cv), fig_path, run_mode, fs=14)

    # Polydegree, lambda plots
    fun.plot_degree_lambda(polydegree, bs_lmb_opt, 'Minimum MSE, Bootstrap',
                           save_bs, fig_path, run_mode)

    fun.plot_degree_lambda(polydegree, cv_lmb_opt, 'Minimum MSE, CV',
                           save_cv, fig_path, run_mode)

    # Heatmaps
    fun.plot_heatmap(lambdas, polydegree, bs_error_test, 'MSE', 'Bootstrap + %s' % reg_str,
                     'bs_error_%s' % save_bs, fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, bs_bias, 'bias', 'Bootstrap + %s' % reg_str,
                     'bs_bias_%s' % save_bs, fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, bs_var, 'var', 'Bootstrap + %s' % reg_str,
                     'bs_variance_%s' % save_bs, fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, cv_error_test, 'MSE', 'CV + %s' % reg_str,
                     'cv_error_%s' % save_cv, fig_path, run_mode)

    # Bias-variance trade-off
    i = 1
    fun.plot_bias_variance(polydegree, bs_error_test_opt[:, i], bs_bias_opt[:, i], bs_var_opt[:, i],
                           '%s, $\lambda$=%.2e, $N$=%d, $N_{bs}$=%d' % (reg_str, bs_best[1], N, N_bootstraps),
                           '%s_opt%d' % (save_bs, i), fig_path, run_mode)

    # MSE train test plots
    fun.plot_MSE_train_test(polydegree, bs_error_train_opt[:, i], bs_error_test_opt[:, i],
                            'p=%d, $\lambda$=%.2e, N=%d, $N_{BS}$=%d' % (bs_best[0], bs_best[1], N, N_bootstraps),
                            '%s_opt%d' % (save_bs, i),
                            fig_path, run_mode, resample='Bootstrap')

    fun.plot_MSE_train_test(polydegree, cv_error_train_opt[:, i], cv_error_test_opt[:, i],
                            'p=%d, $\lambda$=%.2e, N=%d, $N_{BS}$=%d' % (cv_best[0], cv_best[1], N, N_bootstraps),
                            '%s_opt%d' % (save_cv, i),
                            fig_path, run_mode, resample='CV')

    fun.plot_multiple_y(polydegree, [bs_error_test_opt[:, i], cv_error_test_opt[:, i]],
                        ['test, Bootstrap', 'test, CV'],
                        'Comparing bootstrap and cross-validation', 'Polynomial degree', 'Mean squared error',
                        'test_compare_BS_CV_p%d_%s_opt%d' % (p, save_bc, i), fig_path, run_mode)

    # Write bootstrap to file
    fun.save_to_file([bs_error_test_opt[:, 0], bs_bias_opt[:, 0], bs_var_opt[:, 0], bs_lmb_opt],
                     ['bs_error_test', 'bs_bias', 'bs_var', 'bs_lmb'],
                     write_path+'franke/bias_var_task_%s_%s.txt' % (run_mode, save_bs), benchmark)

    # Write CV to file
    fun.save_to_file([cv_error_test_opt[:, 0], cv_error_train[:, 0], cv_lmb_opt],
                     ['cv_error_test', 'cv_error_train', 'cv_lmb'],
                     write_path+'franke/train_test_task_%s_%s.txt' % (run_mode, save_cv), benchmark)

    plt.show()

########################################################################################################################
if run_mode == 'f':
    # fetch terrain data and plot it i guess
    fun.plot_terrain(z_mesh, 'Terrain, location [%d, %d] to [%d, %d]' % (loc_s[0], loc_s[1], loc_e[0], loc_e[1]),
                     'terrain_n%d_%d_%d' % (n_terrain, loc_s[0], loc_s[1]), fig_path, run_mode)

    fun.plot_surf(y_mesh, x_mesh, z_mesh, 'x', 'y', 'z', 'Terrain, $N$=%d' % N,
                  'surf_terrain_n%d_%d_%d' % (n_terrain, loc_s[0], loc_s[1]),
                  fig_path, run_mode, [-0.2, 0.8], azim=-30)

    plt.show()


########################################################################################################################
if run_mode == 'g':

    if reg_str == 'OLS':
        # Setting to 1 for OLS
        nlambdas = 1
        lambdas = np.zeros(1)
    elif reg_str == 'Ridge':
        # Lambdas for Ridge regression
        nlambdas = 15  # 30  # 100
        lambdas = np.logspace(-6, 1, nlambdas)
    elif reg_str == 'Lasso':
        # Lambdas / alphas for Lasso regression
        nlambdas = 15  # 30  # 100
        lambdas = np.logspace(-6, -2, nlambdas)
    max_iter = 500000

    polydegree = np.arange(1, p + 1)
    N_bootstraps = 264
    K = 5

    # Benchmark settings
    if benchmark is True:
        N_bootstraps = 264
        K = 5
        max_iter = 50000
        nlambdas = 5
        lambdas = np.logspace(-4, 0, nlambdas)

    variables = run_regression(X, z, reg_str, polydegree, lambdas, N_bootstraps, K, test_size, scale, max_iter=max_iter)
    # Unpacking variables
    bs_error_train, bs_error_test = variables[0:2]
    bs_bias, bs_var = variables[2:4]
    bs_error_train_opt, bs_error_test_opt = variables[4:6]
    bs_bias_opt, bs_var_opt, bs_lmb_opt = variables[6:9]
    cv_error_train, cv_error_test = variables[9:11]
    cv_error_train_opt, cv_error_test_opt, cv_lmb_opt = variables[11:14]
    bs_min, bs_best, cv_min, cv_best = variables[14:18]
    # way too many

    # Parameters for easier saving to file
    run_mode += '/%s' % reg_str
    save = 'N%d_patch%d_pmax%d_nlamb%d' % (N, patch, p, nlambdas,)
    save_bs = '%s_%s_%s_Nbs%d' % (save, reg_str, 'boot', N_bootstraps)
    save_cv = '%s_%s_%s_k%d' % (save, reg_str, 'cv', K)
    save_bc = '%s_%s_Nbs%d_k%d' % (save, reg_str, N_bootstraps, K)

    # Plotting

    # Lambda, MSE plots
    fun.plot_lambda_mse(lambdas, bs_error_test[bs_min[0], :], '%s, Bootstrap, p=%d' % (reg_str, bs_best[0]),
                        'bootstrap_p%d_%s' % (bs_best[0], save_bs), fig_path, run_mode, fs=14)

    fun.plot_lambda_mse(lambdas, cv_error_test[cv_min[0], :], '%s, Cross-validation, p=%d' % (reg_str, cv_best[0]),
                        'cv_p%d_%s' % (cv_best[0], save_cv), fig_path, run_mode, fs=14)

    # Polydegree, lambda plots
    fun.plot_degree_lambda(polydegree, bs_lmb_opt, 'Minimum MSE, Bootstrap',
                           save_bs, fig_path, run_mode)

    fun.plot_degree_lambda(polydegree, cv_lmb_opt, 'Minimum MSE, CV',
                           save_cv, fig_path, run_mode)

    # Heatmaps
    fun.plot_heatmap(lambdas, polydegree, bs_error_test, 'MSE', 'Bootstrap + %s' % reg_str,
                     'bs_error_%s' % save_bs, fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, bs_bias, 'bias', 'Bootstrap + %s' % reg_str,
                     'bs_bias_%s' % save_bs, fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, bs_var, 'var', 'Bootstrap + %s' % reg_str,
                     'bs_variance_%s' % save_bs, fig_path, run_mode)

    fun.plot_heatmap(lambdas, polydegree, cv_error_test, 'MSE', 'CV + %s' % reg_str,
                     'cv_error_%s' % save_cv, fig_path, run_mode)

    # Bias-variance trade-off
    i = 1
    fun.plot_bias_variance(polydegree, bs_error_test_opt[:, i], bs_bias_opt[:, i], bs_var_opt[:, i],
                           '%s, $\lambda$=%.2e, $N$=%d, $N_{bs}$=%d' % (reg_str, bs_best[1], N, N_bootstraps),
                           '%s_opt%d' % (save_bs, i), fig_path, run_mode)

    # MSE train test plots
    fun.plot_MSE_train_test(polydegree, bs_error_train_opt[:, i], bs_error_test_opt[:, i],
                            'p=%d, $\lambda$=%.2e, N=%d, $N_{BS}$=%d' % (bs_best[0], bs_best[1], N, N_bootstraps),
                            '%s_opt%d' % (save_bs, i),
                            fig_path, run_mode, resample='Bootstrap')

    fun.plot_MSE_train_test(polydegree, cv_error_train_opt[:, i], cv_error_test_opt[:, i],
                            'p=%d, $\lambda$=%.2e, N=%d, $N_{BS}$=%d' % (cv_best[0], cv_best[1], N, N_bootstraps),
                            '%s_opt%d' % (save_cv, i),
                            fig_path, run_mode, resample='CV')

    fun.plot_multiple_y(polydegree, [bs_error_test_opt[:, i], cv_error_test_opt[:, i]],
                        ['test, Bootstrap', 'test, CV'],
                        'Comparing bootstrap and cross-validation', 'Polynomial degree', 'Mean squared error',
                        'test_compare_BS_CV_p%d_%s_opt%d' % (p, save_bc, i), fig_path, run_mode)

    # Write bootstrap to file
    run_mode = 'task_g'  # spaghetti code
    fun.save_to_file([bs_error_test_opt[:, 0], bs_bias_opt[:, 0], bs_var_opt[:, 0], bs_lmb_opt],
                     ['bs_error_test', 'bs_bias', 'bs_var', 'bs_lmb'],
                     write_path+'terrain/bias_var_task_%s_%s.txt' % (run_mode, save_bs), benchmark)

    # Write CV to file
    fun.save_to_file([cv_error_test_opt[:, 0], cv_error_train[:, 0], cv_lmb_opt],
                     ['cv_error_test', 'cv_error_train', 'cv_lmb'],
                     write_path+'terrain/train_test_task_%s_%s.txt' % (run_mode, save_cv), benchmark)

    plt.show()
