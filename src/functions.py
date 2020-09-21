import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

###########################################################


def franke_function(x, y):
    """
    :param x:
    :param y:
    :return:
    """
    term1 = 0.75*np.exp(-0.25*(9*x-2)**2 - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-0.25*(9*x-7)**2 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def mean_squared_error(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


def calculate_R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


def generate_polynomial(x, y, p):
    """
    :param x:
    :param y:
    :param p:
    :return:
    """
    l = polynom_N_terms(p)  # Number of terms in combined polynomial
    X = np.ones((len(x), l))

    j = 0  # Double check, but 99.99% sure it works as intended
    for i in range(1, p + 1):
        j = j + i - 1
        for k in range(i + 1):
            X[:, i + j + k] = x ** (i - k) * y ** k

    return X


def polynom_N_terms(p):
    """
    Returns the amount of terms the polynomial of degree p given by generate_polynomial
    :param p: polynomial degree
    :return:
    """
    return np.sum(np.arange(p+2))


def split_data(X, y, test_size=0.25):
    """
    :param X:
    :param y:
    :param test_size:
    :return:
    """
    N = len(y)
    i_split = int((1-test_size)*N)

    index = np.arange(N)
    np.random.shuffle(index)

    X = X[index]
    y = y[index]

    X_train = X[0:i_split]
    X_test = X[i_split:]

    y_train = y[0:i_split]
    y_test = y[i_split:]

    return X_train, X_test, y_train, y_test


def scale_X(X):
    X_temp = X[:,1:]  # leaving out the intercept
    X_temp -= np.mean(X)
    X[:, 1:] = X_temp

    return X


def invert_SVD(X):
    """
    Computing the pseudo-inverse of X: X^-1 = V s^-1 UT
    :param X: input matrix
    :return: pseudo inverse of input matrix X
    """
    U, s, VT = np.linalg.svd(X)
    inv_sigma = np.zeros(len(s))
    inv_sigma[s != 0] = 1/s[s != 0]  # setting every non-zero element to 1/s[i]
    V = VT.T

    return V @ np.diag(inv_sigma) @ U.T


# TODO: remove, from mortens code
def SVDinv(A):
    ''' Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.
    '''
    U, s, VT = np.linalg.svd(A)
#    print('test U')
#    print( (np.transpose(U) @ U - U @np.transpose(U)))
#    print('test VT')
#    print( (np.transpose(VT) @ VT - VT @np.transpose(VT)))
#    print(U)
 #   print(s)
  #  print(VT)

    print('meow')
    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=s[i]
    UT = np.transpose(U); V = np.transpose(VT); invD = np.linalg.inv(D)
    return np.matmul(V,np.matmul(invD,UT))

###########################################################
################### Plotting functions ####################
###########################################################


def print_MSE_R2(y_data, y_model, data_str, method):
    """
    :param y_data:
    :param y_model:
    :param data_str: 'train', 'test'
    :param method: 'OLS', 'RIDGE'
    :return:
    """
    MSE = mean_squared_error(y_data, y_model)
    R2 = calculate_R2(y_data, y_model)

    data_set = {'train': 'Training', 'test': 'Test'}
    print()  # Newline for readability
    print('%s MSE for %s: %.6f' %(data_set[data_str], method, MSE))
    print('%s R2 for %s: %.6f' %(data_set[data_str], method, R2))
    return


def plot_MSE_train_test(polydegree, train_MSE, test_MSE, n_a, test_size, noise):

    fig = plt.figure()
    plt.plot(polydegree, train_MSE, label='Train')
    plt.plot(polydegree, test_MSE, label='Test')
    plt.legend()
    plt.yscale('log')
    plt.grid('on')
    plt.title('N = %d, test size = %.2f, noise = %.2f' %(n_a, test_size, noise))
#    plt.ylim([0, 0.025])

if __name__ == '__main__':
    pass