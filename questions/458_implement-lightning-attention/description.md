## Problem

Write a Python function `lightning_attention(Q, K, V, decay)` that implements causal linear attention with exponential decay, as used in Lightning Attention. The function takes:
- `Q`: query array of shape `(seq_len, head_dim)`
- `K`: key array of shape `(seq_len, head_dim)`
- `V`: value array of shape `(seq_len, head_dim)`
- `decay`: a float decay factor (lambda) between 0 and 1

Instead of computing softmax(QK^T)V, compute the output using the recurrent form:
- Maintain a state `S` of shape `(head_dim, head_dim)` initialized to zeros
- At each timestep t: `S_t = decay * S_{t-1} + K_t^T @ V_t`, then `O_t = Q_t @ S_t`

Return the output array of shape `(seq_len, head_dim)` as floats. Only use numpy.
