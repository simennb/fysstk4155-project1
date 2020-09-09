import numpy as np

'''
def FrankeFunction(x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4


def create_X(x, y, n ):
        if len(x.shape) > 1:
                x = np.ravel(x)
                y = np.ravel(y)

        N = len(x)
        l = int((n+1)*(n+2)/2)          # Number of elements in beta
        X = np.ones((N,l))

        for i in range(1,n+1):
                q = int((i)*(i+1)/2)
                for k in range(i+1):
                        X[:,q+k] = (x**(i-k))*(y**k)

        return X


# Making meshgrid of datapoints and compute Franke's function
p = 2
n = 5
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
z = FrankeFunction(x, y)
X = create_X(x, y, n=p)
print(X)
'''

p = 6
#print('Morten:')
morten = []
for i in range(1, p + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
                morten.append('X[:,%d] = X**%d * y**%d'%((q+k), (i-k), k))
#                X[:, q + k] = (x ** (i - k)) * (y ** k)


#print('Simen:')
simen = []
j = 0
for i in range(1, p + 1):
        j = j+i-1#i if i % p == 0 else 0
        for k in range(i + 1):
                simen.append('X[:,%d] = X**%d * y**%d'%((i+j+k), (i-k), k))
#                X[:, i + j + k] = x ** (i - k) * y ** k

l = np.sum(range(p+2))  # Number of terms in combined polynomial
meow = []
k = 0
for i in range(1, l + 1):
        k = i%(p)#0 if i % p == 0 else k
        meow.append('X[:,%d] = X**%d * y**%d'%((i), (i-k), k))
        k += 1


for i in range(len(simen)):
        print(morten[i], ' , ', simen[i], ' , ', morten[i]==simen[i])