## Task: Implement the Conjugate Gradient Method for Solving Linear Systems

Your task is to implement the Conjugate Gradient (CG) method, an efficient iterative algorithm for solving large, sparse, symmetric, positive-definite linear systems. Given a matrix `A` and a vector `b`, the algorithm will solve for `x` in the system \( Ax = b \).

Write a function `conjugate_gradient(A, b, n, x0=None, tol=1e-8)` that performs the Conjugate Gradient method as follows:

- `A`: A symmetric, positive-definite matrix representing the linear system.
- `b`: The vector on the right side of the equation.
- `n`: Maximum number of iterations.
- `x0`: Initial guess for the solution vector.
- `tol`: Tolerance for stopping criteria.

The function should return the solution vector `x`.

    
