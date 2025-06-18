
## Understanding Gaussian Elimination

Gaussian Elimination is used to replace matrix coefficients with a row-echelon form matrix, which can be more easily solved via backwards substitution.

### Row-Echelon Form Criteria

- **Non-zero rows** are above any rows of all zeros.
- The **leading entry** of each non-zero row is to the right of the leading entry of the previous row.
- The **leading entry** in any non-zero row is 1, and all entries below it in the same column are zeros.

### Augmented Matrix

For a linear system $Ax = b$, an augmented matrix is a way of displaying all the numerical information in a linear system in a single matrix. This combines the coefficient matrix $A$ and vector source $b$ as follows:

$$
\begin{pmatrix} 
a_{11} & a_{21} & a_{31} & b_1\\ 
a_{12} & a_{22} & a_{32} & b_2\\ 
a_{31} & a_{32} & a_{33} & b_3 
\end{pmatrix}
$$

### Partial Pivoting

In linear algebra, diagonal elements of a matrix are referred to as the "pivot". To solve a linear system, the diagonal is used as a divisor for other elements within the matrix. This means that Gaussian Elimination will fail if there is a zero pivot.

In this case, pivoting is used to interchange rows, ensuring a non-zero pivot. Specifically, **partial pivoting** looks at all other rows in the current column to find the row with the highest absolute value. This row is then interchanged with the current row. This not only increases the numerical stability of the solution, but also reduces round-off errors caused by dividing by small entries.

### Gaussian Elimination Mathematical Formulation

**Gaussian Elimination:**

- For $k = 1$ to $ \text{number of rows} - 1$:
  - Apply partial pivoting to the current row.
  - For $i = k + 1$ to $ \text{number of rows}$:
    - $ m_{ik} = \frac{a_{ik}}{a_{kk}} $
    - For $j = k$ to $ \text{number of columns}$:
      - $ a_{ij} = a_{ij} - m_{ik} \times a_{kj} $
    - $ b_i = b_i - m_{ik} \times b_k $

**Backwards Substitution:**

- For $k = \text{number of rows}$ to $1$:
  - For $i = \text{number of columns} - 1$ to $1$:
    - $ b_k = b_k - a_{ki} \times b_i $
  - $ b_k = \frac{b_k}{a_{kk}} $

### Example Calculation

Let’s solve the system of equations:

$$
A = \begin{pmatrix} 
2 & 8 & 4\\ 
5 & 5 & 1 \\ 
4 & 10 & -1 
\end{pmatrix} 
\quad \text{and} \quad 
b = \begin{pmatrix} 
2 \\ 5 \\ 1 
\end{pmatrix} 
$$

1. Apply **partial pivoting** to increase the magnitude of the pivot. For $A_{11}$, calculate the factor for the elimination of $A_{12}$: 

$$ 
m_{12} = \frac{A_{12}}{A_{11}} = \frac{2}{5} = 0.4 
$$

2. Apply this scaling to row 1 and subtract this from row 2, eliminating $A_{12}$:

$$
A = \begin{pmatrix} 
5 & 5 & 1 \\ 
0 & 6 & 3.6 \\ 
4 & 10 & -1 
\end{pmatrix} 
\quad \text{and} \quad 
b = \begin{pmatrix} 
5 \\ 0 \\ 1 
\end{pmatrix} 
$$

After the full **Gaussian Elimination** process has been applied to $A$ and $b$, we get the following:

$$
A = \begin{pmatrix} 
5 & 5 & 1\\ 
0 & 6 & 3.6 \\ 
0 & 0 & -5.4 
\end{pmatrix} 
\quad \text{and} \quad 
b = \begin{pmatrix} 
5 \\ 0 \\ 3 
\end{pmatrix} 
$$

To calculate $x$, we apply **backward substitution** by substituting in the currently solved values and dividing by the pivot. This gives the following for the first iteration:

$$ 
x_3 = \frac{b_3}{A_{33}} = \frac{3}{-5.4} = -0.56 
$$

This process can be repeated iteratively for all rows to solve the linear system, substituting in the solved values for the rows below.

### Applications

Gaussian Elimination and linear solvers have a wide range of real-world applications, including their use in:

- Machine learning
- Computational fluid dynamics
- 3D graphics
