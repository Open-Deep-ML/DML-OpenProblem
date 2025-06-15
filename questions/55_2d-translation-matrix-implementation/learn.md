
## 2D Translation Matrix Implementation

The translation matrix is a fundamental concept in linear algebra and computer graphics, used to move points or objects in a 2D space.

### Concept Overview

For a 2D translation, we use a 3x3 matrix to move a point \( (x, y) \) by \( x_t \) units in the x-direction and \( y_t \) units in the y-direction.

Any point \( P \) in 2D Cartesian space with coordinates \( (x, y) \) can be represented in homogeneous coordinates as \( (x, y, 1) \):

$$
P_{\text{Cartesian}} = (x, y) \rightarrow P_{\text{Homogeneous}} = (x, y, 1)
$$

More generally, any scalar multiple of \( (x, y, 1) \) represents the same point in 2D space. Thus, \( (kx, ky, k) \) for any non-zero \( k \) also represents the same point \( (x, y) \).

The addition of this third coordinate allows us to represent translation as a linear transformation.

### Translation Matrix

The translation matrix \( T \) is defined as:

$$
T = \begin{bmatrix}
1 & 0 & x_t \\
0 & 1 & y_t \\
0 & 0 & 1
\end{bmatrix}
$$

### Applying the Translation

To translate a point \( (x, y) \), we first convert it to homogeneous coordinates: \( (x, y, 1) \). The transformation is then performed using matrix multiplication:

$$
\begin{bmatrix}
1 & 0 & x_t \\
0 & 1 & y_t \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
=
\begin{bmatrix}
x + x_t \\
y + y_t \\
1
\end{bmatrix}
$$

### Explanation of Parameters

1. **Original Point**: \( (x, y) \)  
2. **Translation in x-direction**: \( x_t \)  
3. **Translation in y-direction**: \( y_t \)  
4. **Translated Point**: \( (x + x_t, y + y_t) \)

This process effectively shifts the original point \( (x, y) \) by \( x_t \) and \( y_t \), resulting in the new coordinates \( (x + x_t, y + y_t) \).
