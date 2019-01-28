import numpy as np
import time

def FUN_SoftThresholding(q, t, K):  
    '''
    The soft-thresholding function returns the optimal x that minimizes 
        min_x 0.5 * x^2 - q * x + t * |x|
    '''
    x = np.maximum(q - t, np.zeros(K)) - np.maximum(-q - t,np.zeros(K));
    return x

def FUN_STELA(N, K, A, y, mu):

    '''
    STELA algorithm solves the following optimization problem:
        min_x 0.5*||y - A * x||^2 + mu * ||x||_1
            
    Reference:
        Sec. IV-C of [Y. Yang, and M. Pesavento, "A unified successive pseudoconvex approximation framework", IEEE Transactions on Signal Processing, 2017]        

    Input Parameters:
        N : the number of measurements
        K : the number of features
        A : N * K,  dictionary matrix
        y : N * 1,  noisy observation
        mu: scalar, regularization gain
        
    Definitions:
        f(x) = 0.5 * ||y - A * x||^2
        g(x) = mu * ||x||_1
        
    Output Parameters:
        x: K * 1, the optimal variable = argmin {f(x) + g(x)}
        objval: objective function value = f + g
        error specifies the solution precision, defined in (53) of the reference
        
    '''
    
    '''precomputation'''
    AtA_diag          = np.sum(np.multiply(A, A), axis = 0) # diagonal elements of A'*A
    mu_vec            = mu * np.ones(K)
    mu_vec_normalized = np.divide(mu_vec, AtA_diag)

    '''initialization'''
    MaxIter  = 100;                     # maximum number of iterations
    x        = np.zeros(K)              # initial point
    objval   = np.zeros(MaxIter + 1)    # objvective function value vs number of iterations
    error    = np.zeros(MaxIter + 1)    # solution precision vs number of iterations    
    CPU_time = np.zeros(MaxIter + 1)    # CPU time (cumulative with respect to iterations)
    
    '''The 0-th iteration'''
    CPU_time[0] = time.time()
    residual    = np.dot(A, x) - y # residual = A * x - y
    f_gradient  = np.dot(residual, A)    # gradient of f
    CPU_time[0] = time.time() - CPU_time[0]
    
    f         = 1/2 * np.dot(residual, residual)
    g         = mu * np.linalg.norm(x, 1)    
    objval[0] = f + g    
    error[0]  = np.linalg.norm(np.absolute(f_gradient - np.minimum(np.maximum(f_gradient - x,-mu * np.ones(K)), mu * np.ones(K))), np.inf) # cf. (53) of reference
#    print("Iteration ", 0, ": function value  ", objval[0], " and error ", error[0])    

    '''formal iterations'''
    for t in range(0 , MaxIter):
        CPU_time[t+1] = time.time()
        
        '''approximate problem, cf. (49) of reference'''    
        Bx = FUN_SoftThresholding(x - np.divide(f_gradient,AtA_diag), mu_vec_normalized, K) 
    
        x_dif  = Bx - x
        Ax_dif = np.dot(A , x_dif) # A * (Bx - x)
    
        '''stepsize, cf. (50) of reference'''
        stepsize_numerator   = -(np.dot(residual,Ax_dif)+np.dot(mu_vec,np.absolute(Bx)-np.absolute(x)))
        stepsize_denominator = np.dot(Ax_dif,Ax_dif)
        stepsize             = np.maximum(np.minimum(stepsize_numerator / stepsize_denominator, 1), 0)

        '''variable update'''
        x = x + stepsize * x_dif;

        residual   = residual + stepsize * Ax_dif;
        f_gradient = np.dot(residual, A)
    
        CPU_time[t+1] = time.time() - CPU_time[t+1] + CPU_time[t]
        
        f            = 1/2 * np.dot(residual, residual)
        g            = mu * np.linalg.norm(x, 1)        
        objval [t+1] = f + g
        error[t+1]   = np.linalg.norm(np.absolute(f_gradient - np.minimum(np.maximum(f_gradient - x, -mu_vec), mu_vec)), np.inf)
        
        '''check stop criterion'''
        if error[t+1] < 1e-6:
            objval    = objval[0 : t + 1]
            CPU_time  = CPU_time[0 : t + 1]
            error     = error[0 : t + 1]
            break
        
#        print("Iteration ", t+1, ": stepsize ", stepsize, ", function value  ", objval[t+1], " error ", error[t+1], " and time ", CPU_time[t+1])    
        
    return objval, x, error
#    return objval
