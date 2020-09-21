import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''
# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)


def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)


def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')
'''

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

np.random.seed(3155)
x = np.random.rand(100)
y = 2.0+5*x*x+0.1*np.random.randn(100)

np.random.seed()
n = 100
mindegree = 0
maxdegree = 14
degrees = range(mindegree, maxdegree)
# Make data set.
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)

X = np.zeros((len(x), len(degrees)))
#    print(x.shape)
for j in degrees:
    X[:, j] = x[:, 0]**j

# We split the data in test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# MSEPredictRidge = np.zeros(len(degrees))
# MSEPredictLasso = np.zeros(len(degrees))

# Ordinary least squares
TrainError = np.zeros(maxdegree)
TestError = np.zeros(maxdegree)
MSETrainOLS = np.zeros(maxdegree)
MSETestOLS = np.zeros(maxdegree)
R2TrainOLS = np.zeros(maxdegree)
R2TestOLS = np.zeros(maxdegree)

# Ridge regression
TrainErrorRidge = np.zeros(maxdegree)
TestErrorRidge = np.zeros(maxdegree)
MSETrainRidge = np.zeros(maxdegree)
MSETestRidge = np.zeros(maxdegree)
R2TrainRidge = np.zeros(maxdegree)
R2TestRidge = np.zeros(maxdegree)

# For iterating over lambdas
nlambdas = 100
lambdas = np.logspace(-4, 0, nlambdas)
optlamb = np.zeros((maxdegree, 4))

# Looping over polynomial degrees
for i in range(len(degrees)):
    deg = degrees[i]
    clf = skl.LinearRegression().fit(X_train_scaled[:,0:i+1], y_train[:,0:i+1])
    y_fit = clf.predict(X_train_scaled[:, 0:i+1])
    y_pred = clf.predict(X_test_scaled[:, 0:i+1])

    # OLS
    TrainError[i] = np.mean( np.mean((y_train - y_fit)**2) )
    TestError[i] = np.mean( np.mean((y_test - y_pred)**2) )
    MSETrainOLS[i] = MSE(y_train, y_fit)
    MSETestOLS[i] = MSE(y_test, y_pred)
    R2TrainOLS[i] = R2(y_train, y_fit)
    R2TestOLS[i] = R2(y_test, y_pred)


    # Ridge racer
    tempMSE_train = np.zeros(nlambdas)
    tempMSE_test = np.zeros(nlambdas)
    tempR2_train = np.zeros(nlambdas)
    tempR2_test = np.zeros(nlambdas)
    for j in range(nlambdas):
        lmb = lambdas[j]
        clf_ridge = skl.Ridge(alpha=lmb).fit(X_train_scaled[:, 0:i + 1], y_train)
        yridgeTrain = clf_ridge.predict(X_train_scaled[:, 0:i + 1])
        yridgeTest = clf_ridge.predict(X_test_scaled[:, 0:i + 1])
        tempMSE_train[j] = MSE(y_train, yridgeTrain)
        tempMSE_test[j] = MSE(y_test, yridgeTest)
        tempR2_train[j] = R2(y_train, yridgeTrain)
        tempR2_test[j] = R2(y_test, yridgeTest)

    MSETrainRidge[i] = min(tempMSE_train)
    MSETestRidge[i] = min(tempMSE_test)
    R2TrainRidge[i] = max(tempR2_train)
    R2TestRidge[i] = max(tempR2_test)

    optlamb[i, 0] = lambdas[np.argmin(tempMSE_train)]
    optlamb[i, 1] = lambdas[np.argmin(tempMSE_test)]
    optlamb[i, 2] = lambdas[np.argmin(1-tempR2_train)]
    optlamb[i, 3] = lambdas[np.argmin(1-tempR2_test)]

    '''
    _lambda = 0.1
    clf_ridge = skl.Ridge(alpha=_lambda).fit(X_train_scaled[:, 0:i+1], y_train)
    yridgeTrain = clf_ridge.predict(X_train_scaled[:, 0:i+1])
    yridgeTest = clf_ridge.predict(X_test_scaled[:, 0:i+1])

    TrainErrorRidge[i] = np.mean( np.mean((y_train - yridgeTrain)**2) )
    TestErrorRidge[i] = np.mean( np.mean((y_test - yridgeTest)**2) )
    MSETrainRidge[i] = MSE(y_train, yridgeTrain)
    MSETestRidge[i] = MSE(y_test, yridgeTest)
    R2TrainRidge[i] = R2(y_train, yridgeTrain)
    R2TestRidge[i] = R2(y_test, yridgeTest)
    '''

'''
# Error
plt.figure()
plt.plot(degrees, TestError, label='Test Error')
plt.plot(degrees, TrainError, label='Train Error')
plt.plot(degrees, TestErrorRidge, '--', label='Test Error Ridge')
plt.plot(degrees, TrainErrorRidge, '--', label='Train Error Ridge')
plt.legend()
'''

# MSE
plt.figure()
plt.plot(degrees, MSETestOLS, label='Test MSE')
plt.plot(degrees, MSETrainOLS, label='Train MSE')
plt.plot(degrees, MSETestRidge, '--', label='Test MSE Ridge')
plt.plot(degrees, MSETrainRidge, '--', label='Train MSE Ridge')
plt.ylabel('MSE')
plt.legend()

# R2
plt.figure()
plt.plot(degrees, R2TestOLS, label='Test R2')
plt.plot(degrees, R2TrainOLS, label='Train R2')
plt.plot(degrees, R2TestRidge, '--', label='Test R2 Ridge')
plt.plot(degrees, R2TrainRidge, '--', label='Train R2 Ridge')
plt.ylabel('R2')
plt.legend()

# Lambdas
plt.figure()
plt.plot(degrees, optlamb[:,0], label='MSE Train')
plt.plot(degrees, optlamb[:,1], label='MSE Test')
plt.plot(degrees, optlamb[:,2], '--', label='R2 Train')
plt.plot(degrees, optlamb[:,3], '--', label='R2 Test')
plt.ylabel('lambda')
plt.legend()




plt.show()




'''
    # Ridge racer
    _lambda = 0.1
    clf_ridge = skl.Ridge(alpha=_lambda).fit(X_train_scaled, y_train)
    yridge = clf_ridge.predict(X_test_scaled)

    # Lasso
    clf_lasso = skl.Lasso(alpha=_lambda).fit(X_train_scaled,y_train)
    ylasso = clf_lasso.predict(X_test_scaled)

    # Compute MSE
    MSEPredictRidge[i] = MSE(y_test, yridge)
    MSEPredictLasso[i] = MSE(y_test, ylasso)

plt.figure()
plt.plot(degrees, MSEPredictRidge, label = 'MSE Ridge')
plt.plot(degrees, MSEPredictLasso, 'r--', label = 'MSE Lasso')
plt.xlabel('Complexity p')
plt.ylabel('MSE')
plt.legend()
plt.show()
'''

# TrainErrorLasso = np.zeros(maxdegree)
# TestErrorLasso = np.zeros(maxdegree)


    # Lasso
#    clf_lasso = skl.Lasso(alpha=_lambda).fit(X_train_scaled[:, 0:i+1],y_train)
#    ylassoTrain = clf_lasso.predict(X_train_scaled[:, 0:i+1])
#    ylassoTest = clf_lasso.predict(X_test_scaled[:, 0:i+1])

#    TrainErrorLasso[i] = np.mean( np.mean((y_train - ylassoTrain)**2) )
#    TestErrorLasso[i] = np.mean( np.mean((y_test - ylassoTest)**2) )

# plt.plot(degrees, TestErrorLasso, label='Test Error Lasso')
# plt.plot(degrees, TrainErrorLasso, label='Train Error Lasso')
