## Learn Section — Deriving an SVD for a 2 x 2 Matrix

The singular-value decomposition (SVD) rewrites any real matrix $A\!\in\!\mathbb R^{m\times n}$ as

$$
A \;=\; U\,\Sigma\,V^{\!\top},
$$

where

- $U\!\in\!\mathbb R^{m\times m}$ and $V\!\in\!\mathbb R^{n\times n}$ are **orthogonal** ($U^{\!\top}U=I$ and $V^{\!\top}V=I$),
- $\Sigma$ is diagonal with **non-negative** entries $\sigma_1\ge\sigma_2\ge\cdots$ called **singular values**.

When $A$ is $2\times2$ the factorisation can be produced **analytically** with only one Givens (Jacobi) rotation.  
Below we walk through the math used in the `svd_2x2_singular_values` function.

---

### 1. From $A$ to a Symmetric Matrix

Because  
$
A^{\!\top}A \;=\; V\,\Sigma^{\!\top}\Sigma\,V^{\!\top}
$  
is symmetric and positive-semidefinite, its eigenvectors form the right-singular vectors of $A$ and its eigenvalues are the squares of the singular values.

For a $2\times2$ matrix

$$
A \;=\; \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix},
\quad
A^{\!\top}A \;=\;
\begin{bmatrix}
a_{11}^2 + a_{21}^2 & a_{11}a_{12}+a_{21}a_{22}\\[4pt]
a_{11}a_{12}+a_{21}a_{22} & a_{12}^2 + a_{22}^2
\end{bmatrix}\!,
$$

which we label $B$ in the code (`a_2`).

---

### 2. A Single Jacobi Rotation

To diagonalise $B$ we seek a rotation matrix

$$
R(\theta)=
\begin{bmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \phantom{-}\cos\theta
\end{bmatrix},
$$

such that $R^{\!\top}BR$ is diagonal.

For $2\times2$ Jacobi iterations the optimal angle is

$$
\theta \;=\;
\begin{cases}
\dfrac{\pi}{4}, & B_{11}=B_{22},\\[8pt]
\dfrac12\,\arctan\!\bigl(\dfrac{2B_{12}}{B_{11}-B_{22}}\bigr), & \text{otherwise}.
\end{cases}
$$

The function computes this `theta`, builds $R$, and updates

$$
D \;=\; R^{\!\top}BR
$$

which now satisfies $D_{12}=D_{21}=0$.

Because we only need *one* rotation to zero the off-diagonal term, the loop runs exactly once.

---

### 3. Extracting Singular Values

The diagonal entries of $D$ are the **eigenvalues** $\lambda_1,\lambda_2$ of $B$.  
Singular values are their square-roots:

$$
\sigma_1 = \sqrt{\lambda_1}, \quad
\sigma_2 = \sqrt{\lambda_2},
\quad
\sigma_1\ge\sigma_2\ge 0.
$$

In code:

```python
s = np.sqrt([d[0,0], d[1,1]])
```

---

### 4. Building $U$

Given $V = R$ (or $V=VR$ after several rotations), the left-singular vectors are

$$
U \;=\; A\,V\,\Sigma^{-1},
$$

where $\Sigma^{-1}=\operatorname{diag}\!\bigl(\tfrac1{\sigma_1},\tfrac1{\sigma_2}\bigr)$.

The multiplication `a @ v @ s_inv` yields an orthogonal $U$.

---

### 5. Putting It All Together

The function finally returns the triple $(U,\; \sigma,\; V^{\!\top})$, giving the exact SVD of $A$:

```python
return (u, s, v.T)
```

---

### Why This Works

* **Eigen-trick** - Diagonalising $A^{\!\top}A$ exposes singular values.  
* **Jacobi rotation** - For $2\times2$ a **single** rotation nulls the off-diagonal term.  
* **Orthonormality** - Both $R$ and $U$ are orthogonal, preserving lengths and angles.

This compact derivation is highly instructive: it shows how SVD generalises *rotation + scaling* in $\mathbb R^2$ and illustrates numerically stable ways to compute the decomposition without heavy linear-algebra libraries.
