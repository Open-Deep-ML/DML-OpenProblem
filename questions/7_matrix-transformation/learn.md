### Matrix Transformation using $T^{-1} A S$

Transforming a matrix $A$ using the operation $T^{-1} A S$ involves several steps. This operation changes the basis of matrix $A$ using two matrices $T$ and $S$, with $T$  being invertible.

### Steps for Transformation

Given matrices $A$, $T$, and $S$:

1. **Check Invertibility**: Verify that $T$ is invertible by ensuring its determinant is non-zero; otherwise, return $-1$.
2. **Compute Inverses**: Find the invers of $T$, denoted as $T^{-1}$.
3. **Perform Matrix Multiplication**: Calculate the transformed matrix:

   $$
   A' = T^{-1} A S
   $$

### Example

If:

$$
A =
\begin{pmatrix} 
1 & 2 \\ 
3 & 4 
\end{pmatrix}
$$

$$
T =
\begin{pmatrix} 
2 & 0 \\ 
0 & 2 
\end{pmatrix}
$$

$$
S =
\begin{pmatrix} 
1 & 1 \\ 
0 & 1 
\end{pmatrix}
$$

#### Check Invertibility:

- $\det(T) = 4 \neq 0$

#### Compute Inverses:

$$
T^{-1} =
\begin{pmatrix} 
\frac{1}{2} & 0 \\ 
0 & \frac{1}{2} 
\end{pmatrix}
$$

#### Perform the Transformation:

$$
A' = T^{-1} A S
$$

$$
A' =
\begin{pmatrix} 
\frac{1}{2} & 0 \\ 
0 & \frac{1}{2} 
\end{pmatrix}
\begin{pmatrix} 
1 & 2 \\ 
3 & 4 
\end{pmatrix}
\begin{pmatrix} 
1 & 1 \\ 
0 & 1 
\end{pmatrix}
$$

$$
A' =
\begin{pmatrix} 
0.5 & 1.5 \\ 
1.5 & 3.5 
\end{pmatrix}
$$
