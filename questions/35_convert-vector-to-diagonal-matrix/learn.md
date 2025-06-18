
## Understanding Diagonal Matrices

A diagonal matrix is a square matrix in which the entries outside the main diagonal are all zero. The main diagonal is the set of entries extending from the top left to the bottom right of the matrix.

### Problem Overview
In this problem, you will write a function to convert a 1D numpy array (vector) into a diagonal matrix. The resulting matrix will have the elements of the input vector on its main diagonal, with zeros elsewhere.

### Mathematical Representation
Given a vector $\mathbf{x} = [x_1, x_2, \ldots, x_n]$, the corresponding diagonal matrix $\mathbf{D}$ is:
$$
\mathbf{D} = \begin{bmatrix}
x_1 & 0 & 0 & \cdots & 0 \\
0 & x_2 & 0 & \cdots & 0 \\
0 & 0 & x_3 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & x_n
\end{bmatrix}
$$

### Importance
Diagonal matrices are important in various mathematical and scientific computations due to their simple structure and useful properties.
