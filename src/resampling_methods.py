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

    def compute(self, N_bs, N_resamples):
        y_pred = np.zeros((self.y_test.shape[0], N_resamples))
        statistic = np.zeros(N_resamples)
#        bias = np.zeros(N_resamples)

        for i in range(N_resamples):
#            X_new, y_new = self.resample(N_bs)
#            X_new, y_new = resample(self.X_train, self.y_train, n_samples=N_bs)
            X_new = self.X_train
            y_new = self.y_train

#            np.random.normal(0, 0.1, 100)

            self.reg.fit(X_new, y_new)
            y_pred[:, i] = self.reg.predict(self.X_test)
            statistic[i] = self.stat(y_pred[:, i], self.y_test)

 #       print(y_pred)

        mean_ = np.average(statistic)
        var_ = np.var(y_pred)
        bias_ = np.mean((self.y_test - np.mean(y_pred, axis=1, keepdims=True)) ** 2)
#        print(bias[i])
#            bias[i] = np.mean((self.y_test - np.mean(y_pred)) ** 2)
#        bias_ = np.mean(bias)

        return mean_, var_, bias_

    def resample(self, N_bs):
        sample_ind = np.random.randint(0, len(self.X_train), N_bs)
        X_new = self.X_train[sample_ind]
        y_new = self.y_train[sample_ind]

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
    def __init__(self):
        pass