## Understanding Cramer's Rule

Cramer's Rule is a method to solve a system of linear equations $Ax = b$ using determinants.

### Requirements
- The coefficient matrix $A$ must be square ($n \times n$).
- The determinant of $A$, $\det(A)$, must be non-zero for a unique solution to exist.

### Formula
For each variable $x_i$, replace the $i$-th column of $A$ with vector $b$ and compute:

$$
x_i = \frac{\det(A_i)}{\det(A)}
$$

Where:
- $A_i$ is the matrix formed by replacing the $i$-th column of $A$ with $b$
- $\det(A)$ is the determinant of the original matrix $A$

### Steps
1. Compute $\det(A)$. If it's 0, return -1.
2. For each variable $x_i$:
   - Replace column $i$ in $A$ with $b$
   - Compute $\det(A_i)$
   - Compute $x_i = \frac{\det(A_i)}{\det(A)}$

### Example
Given:

$$
A = \begin{bmatrix} 2 & -1 & 3 \\ 4 & 2 & 1 \\ -6 & 1 & -2 \end{bmatrix}, \quad b = \begin{bmatrix} 5 \\ 10 \\ -3 \end{bmatrix}
$$

1. $\det(A) = -36.0$
2. Replace each column with $b$:

- $\det(A_1) = -6.0$
- $\det(A_2) = -120.0$
- $\det(A_3) = -96.0$

Then,

$$
x = \left[ \frac{-6}{-36}, \frac{-120}{-36}, \frac{-96}{-36} \right] = [0.1667, 3.3333, 2.6667]
$$

### Applications
- Solving small systems of equations
- Useful in theoretical linear algebra
- Not practical for large matrices due to computational cost
