import numpy as np


class MixedPrecision:
    def __init__(self, loss_scale=1024.0):
        self.loss_scale = loss_scale

    def forward(self, weights, inputs, targets):
        # Convert ALL inputs to float16 for computation (regardless of input dtype)
        weights_fp16 = weights.astype(np.float16)
        inputs_fp16 = inputs.astype(np.float16)
        targets_fp16 = targets.astype(np.float16)

        # Simple forward pass: linear model + MSE loss
        predictions = np.dot(inputs_fp16, weights_fp16)
        loss = np.mean((targets_fp16 - predictions) ** 2)

        # Scale loss and convert back to float32
        scaled_loss = loss.astype(np.float32) * self.loss_scale

        return scaled_loss

    def backward(self, gradients):
        # Convert gradients to float32 for precision (regardless of input dtype)
        gradients_fp32 = gradients.astype(np.float32)

        # Check for overflow (NaN or Inf)
        overflow = np.any(np.isnan(gradients_fp32)) or np.any(np.isinf(gradients_fp32))

        if overflow:
            # Return zero gradients if overflow detected (must be float32)
            return np.zeros_like(gradients_fp32, dtype=np.float32)

        # Unscale gradients (ensure result is float32)
        unscaled_gradients = gradients_fp32 / self.loss_scale
        return unscaled_gradients.astype(np.float32)
