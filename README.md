# STELA
STELA algorithm for sparsity regularized linear regression (LASSO)


STELA algorithm solves the following optimization problem:

    min_x {0.5*||y - A * x||^2 + mu * ||x||_1}
            
            
It is based on the parallel best-response (Jacobi) algorithm with guaranteed convergence. It exhibits a fast, reliable and stable performance.


Reference:
    Sec. IV-C of [Y. Yang, and M. Pesavento, "A unified successive pseudoconvex approximation framework", IEEE Transactions on Signal Processing, 2017]        


Input Parameters:
   
    A :      N * K matrix,  dictionary;
    
    y :      K * 1 vector,  noisy observation;
    
    mu:      scalar, regularization gain;
    
    MaxIter: maximum number of iterations, default = 1000;


Definitions:
    
    f(x) = 0.5 * ||y - A * x||^2;
    
    g(x) = mu * ||x||_1;


Output Parameters:
    
    objval: objective function value = f + g;
    
    x: K * 1 vector, the optimal variable that minimizes {f(x) + g(x)};
    
    error:  specifies the solution precision (a smaller error implies a better solution);
