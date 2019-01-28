# STELA
STELA algorithm for sparsity regularized linear regression (LASSO)

STELA algorithm solves the following optimization problem:
    min_x $0.5*||y - A * x||^2 + mu * ||x||_1$
            
It is based on the parallel best-response (Jacobi) algorithm with guaranteed convergence. It exhibits a fast, reliable and stable performance.

Reference:
    Sec. IV-C of [Y. Yang, and M. Pesavento, "A unified successive pseudoconvex approximation framework", IEEE Transactions on Signal Processing, 2017]        

Input Parameters:
    N : number of measurements
    K : number of features
    A : N * K,  dictionary matrix
    y : K * 1,  noisy observation
    mu: scalar, regularization gain
        
Definitions:
    f(x) = 0.5 * ||y - A * x||^2
    g(x) = mu * ||x||_1
        
Output Parameters:
    objval: objective function value = f + g
    x: K * 1, the optimal variable = argmin {f(x) + g(x)}
    error specifies the solution precision, defined in (53) of the reference
