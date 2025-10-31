## Problem

Write a Python function `checkpoint_forward` that takes a list of numpy functions (each representing a layer or operation) and an input numpy array, and returns the final output by applying each function in sequence. To simulate gradient checkpointing, the function should not store intermediate activations; instead, it should recompute them as needed (for this problem, just apply the functions in sequence as usual). Only use standard Python and numpy. The returned array should be of type float and have the same shape as the output of the last function.
