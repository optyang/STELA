import numpy as np
import matplotlib.pyplot as plt
from SelfDefinedFunctions import FUN_STELA

N = 1000; # number of rows of A (measurements)
K = 2000; # number of columns of A (features)

'''disable to input your own number of measurements and features'''
#N = int(input("Please enter the number of measurements: "))
#K = int(input("Please enter the number of features: "))

'''generate measurements: y = A * x0 + v'''
A = np.random.normal(0, 0.1, (N, K))
## =============================================================================
## normalize each row of A
#A_row_norm = np.linalg.norm(A, axis=1)
#A_row_norm_matrix = np.matrix.transpose(np.kron(A_row_norm,np.ones((K,1))))
#A = np.divide(A, A_row_norm_matrix)
## =============================================================================

'''generate the sparse vector'''
density          = 0.01 # density of the sparse signal
x0               = np.zeros(K)
x0_positions     = np.random.choice(np.arange(K), int(K * density), replace = False)
x0[x0_positions] = np.random.normal(0, 1, int(K * density))

'''generate the noise'''
sigma = 0.01; # noise standard deviation
v     = np.random.normal(0, sigma, N) # noise

'''generate the noisy measurement'''
y  = np.dot(A, x0) + v # measurement

'''regularization gain'''
mu = 0.01*np.linalg.norm(np.dot(y, A), np.inf)

'''call STELA'''
MaxIter = 100 # maximum number of iterations, optional input
objval, x, error = FUN_STELA(A, y, mu, MaxIter)

'''plot output'''
'''compare the original signal and the estimated signal'''
plt.plot(np.linspace(1, K, K), x0, 'b-x', label = "original signal")
plt.plot(np.linspace(1, K, K), x, 'r-o', label = "estimated signal")
plt.legend(bbox_to_anchor = (1.05, 1), loc = 1, borderaxespad = 0.)
plt.xlabel("index")
plt.ylabel("coefficient")
plt.show()

'''number of iterations vs. objective function value'''
plt.plot(np.linspace(0, objval.size-1, objval.size), objval, 'r-')
plt.xlabel("number of iterations")
plt.ylabel("objective function value")
plt.show()

'''number of iterations vs. solution precision'''
plt.plot(np.linspace(0, error.size-1, error.size), error, 'r-')
plt.xlabel("number of iterations")
plt.ylabel("error")
plt.yscale('log')
plt.show()
