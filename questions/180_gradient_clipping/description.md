## Problem

Write a Python function `clip_gradients` that takes a numpy array of gradients and a float `max_norm`, and returns a new numpy array where the gradients are clipped so that their L2 norm does not exceed `max_norm`. If the L2 norm of the input gradients is less than or equal to `max_norm`, return the gradients unchanged. If it exceeds `max_norm`, scale all gradients so that their L2 norm equals `max_norm`. Only use standard Python and numpy. The returned array should be of type float and have the same shape as the input.
