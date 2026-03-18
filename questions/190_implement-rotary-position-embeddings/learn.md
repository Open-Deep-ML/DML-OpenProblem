# **Rotary Position Embeddings (RoPE)**

## **1. Definition**
Rotary Position Embeddings (RoPE) encode positional information by rotating query and key vectors in the attention mechanism. Unlike absolute or learned positional embeddings added to token embeddings, RoPE applies a position-dependent rotation directly to Q and K, making the dot product between them naturally encode relative position.

## **2. Core Idea**
RoPE groups the dimensions of each head into pairs $(x_{2i}, x_{2i+1})$ and rotates each pair by an angle that depends on the position and frequency:

$$
\theta_i = \text{base}^{-2i/d}
$$

where $d$ is the rotary dimension, $i \in \{0, 1, \ldots, d/2 - 1\}$, and base is typically 10000.

For a token at position $m$, the angle for pair $i$ is:

$$
\alpha_{m,i} = m \cdot \theta_i
$$

## **3. Rotation Formula**
Each pair $(x_{2i}, x_{2i+1})$ is rotated by angle $\alpha_{m,i}$:

$$
\begin{bmatrix} x'_{2i} \\ x'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(\alpha_{m,i}) & -\sin(\alpha_{m,i}) \\ \sin(\alpha_{m,i}) & \cos(\alpha_{m,i}) \end{bmatrix} \begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix}
$$

Which expands to:
$$
x'_{2i} = x_{2i} \cos(\alpha_{m,i}) - x_{2i+1} \sin(\alpha_{m,i})
$$
$$
x'_{2i+1} = x_{2i} \sin(\alpha_{m,i}) + x_{2i+1} \cos(\alpha_{m,i})
$$

## **4. Partial RoPE**
In architectures like MiniMax M2.5, RoPE is applied to only a subset of the head dimensions (e.g., first 64 of 128). The remaining dimensions pass through unchanged. This is called **partial RoPE**.

## **5. Efficient Implementation**
A common implementation trick uses the `rotate_half` function:
1. Split $x$ into first half $x_1$ and second half $x_2$
2. Compute `rotate_half(x) = [-x_2, x_1]`
3. Apply: `x_rotated = x * cos + rotate_half(x) * sin`

## **6. Key Properties**
- **Relative position encoding:** The dot product $q_m^T k_n$ depends only on $m - n$
- **No learnable parameters:** The rotation angles are deterministic
- **Extrapolation:** Can handle longer sequences than seen during training (with appropriate base scaling)

## **7. Role in MiniMax M2.5**
MiniMax M2.5 uses RoPE with:
- Base $\theta = 5{,}000{,}000$ (very large for long context)
- Rotary dimension: 64 out of 128 head dimensions (partial RoPE)
- Applied to Q and K after QK-normalization
