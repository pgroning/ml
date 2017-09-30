import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt

def f(params):
    a, b, c = params
    y = np.abs(a**2 + b**2 + c**2 - 3)
    return y

initial_guess = [1, 1, 1]
result = optimize.minimize(f, initial_guess, method="L-BFGS-B",
                           bounds=[[0,2], [0,2], [0,2]])
a, b, c = result.x
#print a, b, c
#print "y=" + str(f(result.x))
#print result


def model(t, params):
    a, b, c = params
    y = a + b*t + c*t**2
    return y

def optfun(params, y, t):
    #return np.sum(np.abs((y - model(t, params))/y))
    return np.sum((y-model(t, params))**2)

y = np.array([1, 0.1, -3.1, -1, 1.1])
t = np.arange(len(y))

x0 = [1.5, -3, 1]

res = optimize.minimize(optfun, x0, args=(y, t),
                        method='L-BFGS-B', bounds=[[None, None], [None, None],
                                                   [None, None]])
print res

ti = np.linspace(0, 4, 100)
plt.plot(t, y, 'bo', ti, model(ti, res.x), 'r')
plt.show()

