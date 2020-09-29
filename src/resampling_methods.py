import numpy as np
from sklearn.utils import resample


class Bootstrap:
    def __init__(self, X_train, X_test, y_train, y_test, reg_obj, stat):
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
        self.stat = stat

    def compute(self, N_bootstraps, N_samples=None):
        y_pred = np.zeros((self.y_test.shape[0], N_bootstraps))
        y_test = self.y_test.reshape(-1, 1)
        statistic = np.zeros(N_bootstraps)
#        bias = np.zeros(N_resamples)

        tot_unique = np.zeros(N_bootstraps)
        n_samp = np.zeros(N_bootstraps)
        for i in range(N_bootstraps):
#            print(1)
            X_new, y_new = self.resample(self.X_train, self.y_train)
#            X_new, y_new = resample(self.X_train, self.y_train)#, n_samples=N_bs)
            tot_unique[i] = len(np.unique(y_new))
            n_samp[i] = len(X_new[:, -1])
#            X_new = self.X_train
#            y_new = self.y_train

#            np.random.normal(0, 0.1, 100)

            self.reg.fit(X_new, y_new)
            y_pred[:, i] = self.reg.predict(self.X_test)
#            statistic[i] = self.stat(y_pred[:, i], y_test)


#        error[degree] = np.mean(np.mean((y_test - y_pred) ** 2, axis=1, keepdims=True))
#        bias[degree] = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True)) ** 2)
#        variance[degree] = np.mean(np.var(y_pred, axis=1, keepdims=True))

#            print(len(np.unique(y_pred[:, i])))

 #       print(y_pred)
        print('N_samples = %.2f , mean(unique) = %.2f  BS' % (np.mean(n_samp), np.mean(tot_unique)))

        error = np.mean(np.mean((y_test - y_pred) ** 2, axis=1, keepdims=True))
        #error = np.mean(statistic)
        variance = np.mean(np.var(y_pred, axis=1, keepdims=True))
        bias = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True)) ** 2)
#        print(bias[i])
#            bias[i] = np.mean((self.y_test - np.mean(y_pred)) ** 2)
#        bias_ = np.mean(bias)

        return error, variance, bias  # mean_, var_, bias_

    def resample(self, X, y):
        sample_ind = np.random.randint(0, len(X), len(X))

        X_new = (X[sample_ind]).copy()
        y_new = (y[sample_ind]).copy()

        return X_new, y_new


'''
from numpy import *
from numpy.random import randint, randn
from time import time
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Returns mean of bootstrap samples
def stat(data):
    return mean(data)

# Bootstrap algorithm
def bootstrap(data, statistic, R):
    t = zeros(R); n = len(data); inds = arange(n); t0 = time()
    # non-parametric bootstrap
    for i in range(R):
        t[i] = statistic(data[randint(0,n,n)])

    # analysis
    print("Runtime: %g sec" % (time()-t0)); print("Bootstrap Statistics :")
    print("original           bias      std. error")
    print("%8g %8g %14g %15g" % (statistic(data), std(data),mean(t),std(t)))
    return t


mu, sigma = 100, 15
datapoints = 10000
x = mu + sigma*random.randn(datapoints)
# bootstrap returns the data sample
t = bootstrap(x, stat, datapoints)
# the histogram of the bootstrapped  data
n, binsboot, patches = plt.hist(t, 50, normed=1, facecolor='red', alpha=0.75)

# add a 'best fit' line
y = mlab.normpdf( binsboot, mean(t), std(t))
lt = plt.plot(binsboot, y, 'r--', linewidth=1)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.axis([99.5, 100.6, 0, 3.0])
plt.grid(True)

plt.show()
'''


class CrossValidation:
    def __init__(self, X, y, reg_obj, stat):
        """
        :param X:
        :param y:
        :param reg_obj:
#        :param stat: function to compute some statistic
        """
        self.X = X
        self.y = y
        self.reg = reg_obj
        self.stat = stat

    def compute(self, K):
        error = np.zeros(K)

        # TODO: Don't shuffle
#        index = np.arange(len(self.y))
#        np.random.shuffle(index)
#        index = np.arange(len(self.X))
#        np.random.shuffle(index)
#        X = (self.X[index]).copy()
#        y = (self.y[index]).copy()

        X = self.X.copy()
        y = self.y.copy()

        for i in range(K):
            X_train, X_test, y_train, y_test = self.split(X, y, K, i)

            self.reg.fit(X_train, y_train)
            y_pred = self.reg.predict(X_test)
            error[i] = np.mean((y_test - y_pred) ** 2)

        return np.mean(error)

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
        print(X_train.shape, X_test.shape)
#        print(test_indices)
#        print('train ', X_train)
#        print('test ', X_test)

        y_train = y[train_indices]
        y_test = y[test_indices]

        return X_train, X_test, y_train, y_test