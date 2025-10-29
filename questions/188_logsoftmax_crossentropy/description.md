# Log-Softmax and Cross-Entropy Loss Implementation

Implement a numerically stable log-softmax function and use it to compute the cross-entropy loss from scratch in PyTorch.

You are **not allowed** to use `torch.nn.functional.log_softmax` or `torch.nn.functional.cross_entropy`. You may only use basic PyTorch tensor operations.

Your function should support:
- 1D or 2D input tensors
- Batch computation
- Numerical stability using the log-sum-exp trick
