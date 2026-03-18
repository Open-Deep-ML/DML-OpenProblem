## Problem

Write a Python function `swiglu_ffn(x, W1, W3, W2)` that implements the SwiGLU Feed-Forward Network used in modern transformer architectures. The function takes:
- `x`: input array of shape `(batch_size, hidden_dim)`
- `W1`: gate projection weight of shape `(intermediate_dim, hidden_dim)`
- `W3`: up projection weight of shape `(intermediate_dim, hidden_dim)`
- `W2`: down projection weight of shape `(hidden_dim, intermediate_dim)`

Compute: `output = W2^T @ (SiLU(W1^T @ x) * (W3^T @ x))` where SiLU(z) = z * sigmoid(z). Return the output as floats. Only use numpy.
