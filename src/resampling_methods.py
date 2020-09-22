import numpy as np


class Bootstrap:
    def __init__(self, X_train, X_test, y_train, y_test, reg_obj, stat):
        """
        :param X_init:
        :param y_init:
        :param X_test:
        :param y_test:
        :param reg_obj:
        :param stat: list of functions to compute some statistic [func1, func2]
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.reg = reg_obj
        self.stat = stat
        self.N_stat = len(self.stat)

    def compute(self, N_bs, N_resamples):
        statistic = np.zeros((N_resamples, self.N_stat))

        for i in range(N_resamples):
            X_new, y_new = self.resample(N_bs)
            self.reg.fit(X_new, y_new)
            y_predict = self.reg.predict(self.X_test)

            for j in range(self.N_stat):
                statistic[i, j] = self.stat[j](y_predict, self.y_test)

        mean_, var_ = self.compute_mean_variance(statistic)

        return mean_, var_

    def resample(self, N_bs):
        sample_ind = np.random.randint(0, len(self.X_train), N_bs)
        X_new = self.X_train[sample_ind]
        y_new = self.y_train[sample_ind]

        return X_new, y_new

    def compute_mean_variance(self, statistic):
        mean_ = np.zeros(self.N_stat)
        var_ = np.zeros(self.N_stat)
        for i in range(self.N_stat):
            mean_[i] = np.average(statistic[:, i])
            var_[i] = np.var(statistic[:, i])

        return mean_, var_


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
    def __init__(self):
        pass