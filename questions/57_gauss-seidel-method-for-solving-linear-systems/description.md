## Task: Implement the Gauss-Seidel Method

Your task is to implement the Gauss-Seidel method, an iterative technique for solving a system of linear equations \(Ax = b\).

The function should iteratively update the solution vector \(x\) by using the most recent values available during the iteration process.

Write a function `gauss_seidel(A, b, n, x_ini=None)` where:

- `A` is a square matrix of coefficients,
- `b` is the right-hand side vector,
- `n` is the number of iterations,
- `x_ini` is an optional initial guess for \(x\) (if not provided, assume a vector of zeros).

The function should return the approximated solution vector \(x\) after performing the specified number of iterations.

    
