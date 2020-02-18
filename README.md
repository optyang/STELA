# STELA
STELA algorithm for sparsity regularized linear regression (LASSO)


STELA algorithm solves the following optimization problem:

![equation](https://latex.codecogs.com/svg.latex?\dpi{300}&space;\min_{x}&space;\frac{1}{2}\Vert&space;y&space;-&space;Ax\Vert_2^2&space;&plus;&space;\mu\Vert&space;x&space;\Vert_1)
            
It is based on the parallel best-response (Jacobi) algorithm with guaranteed convergence. It exhibits a fast, reliable and stable performance.


Reference: Sec. IV-C of
     Y. Yang, and M. Pesavento, "A unified successive pseudoconvex approximation framework", *IEEE Transactions on Signal Processing*, vol. 65, no. 13, pp. 3313-3328, Jul. 2017. URL: [IEEE](https://ieeexplore.ieee.org/document/7882695), [Arxiv](https://arxiv.org/abs/1506.04972)


Input Parameters:
   
    A :      N * K matrix,  dictionary;
    
    y :      K * 1 vector,  noisy observation;
    
    mu:      scalar, regularization gain;
    
    MaxIter: maximum number of iterations, default = 1000;


Definitions:

![equation](https://latex.codecogs.com/svg.latex?\dpi{300}&space;f(x)&space;=&space;\frac{1}{2}\Vert&space;y&space;-&space;A&space;x\Vert^2)
    
![equation](https://latex.codecogs.com/svg.latex?\dpi{300}&space;g(x)&space;=&space;\mu&space;\Vert&space;x&space;\Vert_1)


Output Parameters:
    
    objval: objective function value (f + g);
    
    x:      K * 1 vector, the optimal variable
    
    error:  specifies the solution precision (a smaller error implies a better solution);
