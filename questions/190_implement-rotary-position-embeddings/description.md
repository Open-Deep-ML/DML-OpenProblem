## Problem

Write a Python function `apply_rotary_embeddings(q, k, positions, dim, base)` that applies Rotary Position Embeddings (RoPE) to query and key tensors. The function takes query `q` and key `k` arrays of shape `(seq_len, head_dim)`, a 1D integer array `positions` of length `seq_len`, the rotary dimension `dim` (number of dimensions to rotate, must be even and <= head_dim), and the `base` frequency (default 10000.0). Return a tuple of the rotated `(q, k)` arrays as floats. Only use numpy.
