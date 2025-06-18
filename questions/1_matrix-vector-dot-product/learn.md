
## Matrix-Vector Dot Product

Consider a matrix $A$ and a vector $v$:

**Matrix $A$ (n x m):**
$$
A = \begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1m} \\
a_{21} & a_{22} & \cdots & a_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nm}
\end{pmatrix}
$$

**Vector $v$ (length m):**
$$
v = \begin{pmatrix}
v_1 \\
v_2 \\
\vdots \\
v_m
\end{pmatrix}
$$

The dot product $A \cdot v$ produces a new vector of length $n$:
$$
A \cdot v = \begin{pmatrix}
a_{11}v_1 + a_{12}v_2 + \cdots + a_{1m}v_m \\
a_{21}v_1 + a_{22}v_2 + \cdots + a_{2m}v_m \\
\vdots \\
a_{n1}v_1 + a_{n2}v_2 + \cdots + a_{nm}v_m
\end{pmatrix}
$$

### Key Requirement:
The number of columns in the matrix ($m$) must equal the length of the vector ($m$). If not, the operation is undefined, and the function should return -1.
