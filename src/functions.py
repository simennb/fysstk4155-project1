import os
import numpy as np

###########################################################


def franke_function(x, y):
    '''
    :param x:
    :param y:
    :return:
    '''
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def mean_squared_error(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


def generate_polynomial(n, p):
    '''
    :param n:
    :param p:
    :return:
    '''
    X = np.zeros((len(x), p))

    X[:, 0] = 1
    X[:, 1] = x
    X[:, 2] = x ** 2

###########################################################
################### Plotting functions ####################
###########################################################


if __name__ == '__main__':
    pass