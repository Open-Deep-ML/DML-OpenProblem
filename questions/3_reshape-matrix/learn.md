
## Reshaping a Matrix

Matrix reshaping involves changing the shape of a matrix without altering its data. This is essential in many machine learning tasks where the input data needs to be formatted in a specific way.

For example, consider a matrix $M$:

**Original Matrix $M$:**
$$
M = \begin{pmatrix} 
1 & 2 & 3 & 4 \\ 
5 & 6 & 7 & 8 
\end{pmatrix}
$$

**Reshaped Matrix $M'$ with shape (4, 2):**
$$
M' = \begin{pmatrix} 
1 & 2 \\ 
3 & 4 \\ 
5 & 6 \\ 
7 & 8 
\end{pmatrix}
$$

### Important Note:
Ensure the total number of elements remains constant during reshaping.
