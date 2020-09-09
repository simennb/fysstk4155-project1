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


def generate_polynomial(x, y, p):
    '''
    :param x:
    :param y:
    :param p:
    :return:
    '''
    l = np.sum(np.range(p+2))  # Number of terms in combined polynomial
    X = np.ones((len(x), l))

    j = 0  # Double check, but 99.99% sure it works as intended
    for i in range(1, p + 1):
        j = j + i - 1
        for k in range(i + 1):
                X[:, i + j + k] = x ** (i - k) * y ** k

    return X

###########################################################
################### Plotting functions ####################
###########################################################


if __name__ == '__main__':
    pass