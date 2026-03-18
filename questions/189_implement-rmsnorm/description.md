## Problem

Write a Python function `rmsnorm(x, weight, eps)` that implements Root Mean Square Layer Normalization. The function takes a 2D numpy array `x` of shape `(batch_size, features)`, a 1D numpy array `weight` of shape `(features,)` for the learnable scale parameter, and a small float `eps` for numerical stability. Return the normalized array as floats. Only use numpy.
