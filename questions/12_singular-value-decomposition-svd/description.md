Write a Python function called svd_2x2_singular_values(A) that finds an approximate singular value decomposition of a real 2 x 2 matrix using one Jacobi rotation.
Input
A: a NumPy array of shape (2, 2)

Rules
You may use basic NumPy operations (matrix multiplication, transpose, element wise math, etc.).
Do not call numpy.linalg.svd or any other high-level SVD routine.
Stick to a single Jacobi step no iterative refinements.

Return
A tuple (U, Σ, V_T) where
U is a 2 x 2 orthogonal matrix,
Σ is a length 2 NumPy array containing the singular values, and
V_T is the transpose of the right-singular-vector matrix V.
