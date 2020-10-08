import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import KFold
import sys
import functions as fun


class Bootstrap:
    def __init__(self, X_train, X_test, y_train, y_test, reg_obj):
        """
        :param X_init:
        :param y_init:
        :param X_test:
        :param y_test:
        :param reg_obj:
#        :param stat: function to compute some statistic
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.reg = reg_obj

    def compute(self, N_bootstraps):
        """
        :param N_bootstraps:
        :param N_samples:
        :return:
        """
        y_pred = np.zeros((self.y_test.shape[0], N_bootstraps))
        y_fit = np.zeros((self.y_train.shape[0], N_bootstraps))
        y_test = self.y_test.reshape(-1, 1)

        train_error = 0
        for i in range(N_bootstraps):
            X_new, y_new = self.resample(self.X_train, self.y_train)
#            X_new, y_new = resample(self.X_train, self.y_train)

            self.reg.fit(X_new, y_new)
            y_pred[:, i] = self.reg.predict(self.X_test)
            y_fit[:, i] = self.reg.predict(X_new)
            train_error += np.mean((y_new - y_fit[:, i])**2) / N_bootstraps

        error = np.mean(np.mean((y_test - y_pred) ** 2, axis=1, keepdims=True))
        bias = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True)) ** 2)
        variance = np.mean(np.var(y_pred, axis=1, keepdims=True))

        return error, bias, variance, train_error

    def resample(self, X, y):
        sample_ind = np.random.randint(0, len(X), len(X))

        X_new = (X[sample_ind]).copy()
        y_new = (y[sample_ind]).copy()

        return X_new, y_new


class CrossValidation:
    def __init__(self, X, y, reg_obj):
        """
        :param X:
        :param y:
        :param reg_obj:
        """
        self.X = X
        self.y = y
        self.reg = reg_obj

    def compute(self, K):
        error_train = np.zeros(K)
        error_test = np.zeros(K)

        # TODO: Don't shuffle
        index = np.arange(len(self.y))
        np.random.shuffle(index)
#        index = np.arange(len(self.X))
#        np.random.shuffle(index)
        X = (self.X[index]).copy()
        y = (self.y[index]).copy()

#        X = self.X.copy()
#        y = self.y.copy()

        for i in range(K):
            X_train, X_test, y_train, y_test = self.split(X, y, K, i)

            self.reg.fit(X_train, y_train)
            y_fit = self.reg.predict(X_train)
            y_pred = self.reg.predict(X_test)

            error_train[i] = np.mean((y_train - y_fit) ** 2)
            error_test[i] = np.mean((y_test - y_pred) ** 2)

        return np.mean(error_train), np.mean(error_test)

    def split(self, X, y, K, i):
        # TODO: Check behavior compared to SKL kfold.split
        # TODO: since not every number is neatly divisable with K

        N = len(X)
#        j = i*int(N/K)
#        l = j + int(N/K)
        j = int(i*N/K)  # TODO: check
        l = j + int(N/K)

#        print('Split #%d: N: %d, [%d : %d], %d' % (i, N, j, l, l-j))

        indices = np.arange(N)
        test_indices = indices[j:l]
        mask = np.ones(N, dtype=bool)
        mask[test_indices] = False
        train_indices = indices[mask]

        X_train = X[train_indices]
        X_test = X[test_indices]
#        print(X_train.shape, X_test.shape)
#        print(test_indices)
#        print('train ', X_train)
#        print('test ', X_test)

        y_train = y[train_indices]
        y_test = y[test_indices]

        return X_train, X_test, y_train, y_test

    def split_2(self, X, y, K):
        # TODO: Check behavior compared to SKL kfold.split
        # TODO: since not every number is neatly divisable with K
        N = len(X)
#        train_indices = np.zeros((N,), dtype=int)

        for i in range(K):

#        j = i*int(N/K)
#        l = j + int(N/K)
            j = int(i*N/K)  # TODO: check
            l = j + int(N/K)

#        print('Split #%d: N: %d, [%d : %d], %d' % (i, N, j, l, l-j))

            indices = np.arange(N)
            test_indices = indices[j:l]
            mask = np.ones(N, dtype=bool)
            mask[test_indices] = False
            train_indices = indices[mask]

        X_train = X[train_indices]
        X_test = X[test_indices]
#        print(X_train.shape, X_test.shape)
#        print(test_indices)
#        print('train ', X_train)
#        print('test ', X_test)

        y_train = y[train_indices]
        y_test = y[test_indices]

        return X_train, X_test, y_train, y_test



class CrossValidationSKL:
    def __init__(self, X, y, reg_obj):
        """
        :param X:
        :param y:
        :param reg_obj:
        """
        self.X = X
        self.y = y
        self.reg = reg_obj

    def compute(self, K):
#        kfold = KFold(n_splits=K)
        kfold = KFold(n_splits=K, random_state=0, shuffle=True)
        error_train = 0
        error_test = 0
        for train_inds, test_inds in kfold.split(self.X):
            X_train = self.X[train_inds]
            y_train = self.y[train_inds]

            X_test = self.X[test_inds]
            y_test = self.y[test_inds]

            beta = self.reg.fit(X_train, y_train)
            y_fit = self.reg.predict(X_train)
            y_predict = self.reg.predict(X_test)

            error_train += fun.mean_squared_error(y_train, y_fit) / K
            error_test += fun.mean_squared_error(y_test, y_predict) / K

        return error_train, error_test
