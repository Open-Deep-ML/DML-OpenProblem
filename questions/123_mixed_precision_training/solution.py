import numpy as np
import json

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
        loss = np.mean((predictions - targets_fp16) ** 2)
        
        # Scale loss and convert back to float32 (Python float)
        scaled_loss = float(loss) * self.loss_scale
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
    

tests = [
  {
    "input": "import numpy as np\nmp = MixedPrecision(loss_scale=1024.0)\nweights = np.array([0.5, -0.3], dtype=np.float32)\ninputs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)\ntargets = np.array([1.0, 0.0], dtype=np.float32)\nloss = mp.forward(weights, inputs, targets)\nprint(f\"Loss: {loss:.4f}\")\nprint(f\"Loss dtype: {type(loss).__name__}\")\ngrads = np.array([512.0, -256.0], dtype=np.float32)\nresult = mp.backward(grads)\nprint(f\"Gradients: {result}\")\nprint(f\"Grad dtype: {result.dtype}\")",
    "output": "Loss: 1638.4000\nLoss dtype: float\nGradients: [0.5 -0.25]\nGrad dtype: float32"
  },
  {
    "input": "import numpy as np\nmp = MixedPrecision(loss_scale=512.0)\nweights = np.array([1.0, 0.5], dtype=np.float64)\ninputs = np.array([[2.0, 1.0]], dtype=np.float64)\ntargets = np.array([3.0], dtype=np.float64)\nloss = mp.forward(weights, inputs, targets)\nprint(f\"Loss: {loss:.1f}\")\nprint(f\"Loss dtype: {type(loss).__name__}\")\ngrads = np.array([1024.0, 512.0], dtype=np.float16)\nresult = mp.backward(grads)\nprint(f\"Gradients: [{result[0]:.0f} {result[1]:.0f}]\")\nprint(f\"Grad dtype: {result.dtype}\")",
    "output": "Loss: 256.0\nLoss dtype: float\nGradients: [2 1]\nGrad dtype: float32"
  },
  {
    "input": "import numpy as np\nmp = MixedPrecision(loss_scale=100.0)\nweights = np.array([0.1, 0.2], dtype=np.float32)\ninputs = np.array([[1.0, 1.0]], dtype=np.float32)\ntargets = np.array([0.5], dtype=np.float32)\nloss = mp.forward(weights, inputs, targets)\nprint(f\"Loss: {loss:.1f}\")\nprint(f\"Loss dtype: {type(loss).__name__}\")\ngrads = np.array([200.0, 100.0], dtype=np.float64)\nresult = mp.backward(grads)\nprint(f\"Gradients: [{result[0]:.0f} {result[1]:.0f}]\")\nprint(f\"Grad dtype: {result.dtype}\")",
    "output": "Loss: 4.0\nLoss dtype: float\nGradients: [2 1]\nGrad dtype: float32"
  },
  {
    "input": "import numpy as np\nmp = MixedPrecision(loss_scale=2048.0)\nweights = np.array([0.25], dtype=np.float64)\ninputs = np.array([[4.0]], dtype=np.float64)\ntargets = np.array([2.0], dtype=np.float64)\nloss = mp.forward(weights, inputs, targets)\nprint(f\"Loss: {loss:.1f}\")\nprint(f\"Loss dtype: {type(loss).__name__}\")\ngrads = np.array([np.nan], dtype=np.float16)\nresult = mp.backward(grads)\nprint(f\"Gradients: [{result[0]:.0f}]\")\nprint(f\"Grad dtype: {result.dtype}\")",
    "output": "Loss: 2048.0\nLoss dtype: float\nGradients: [0]\nGrad dtype: float32"
  },
  {
    "input": "import numpy as np\nmp = MixedPrecision(loss_scale=256.0)\nweights = np.array([1.0], dtype=np.float16)\ninputs = np.array([[2.0]], dtype=np.float16)\ntargets = np.array([3.0], dtype=np.float16)\nloss = mp.forward(weights, inputs, targets)\nprint(f\"Loss: {loss:.1f}\")\nprint(f\"Loss dtype: {type(loss).__name__}\")\ngrads = np.array([np.inf], dtype=np.float64)\nresult = mp.backward(grads)\nprint(f\"Gradients: [{result[0]:.0f}]\")\nprint(f\"Grad dtype: {result.dtype}\")",
    "output": "Loss: 256.0\nLoss dtype: float\nGradients: [0]\nGrad dtype: float32"
  }
]

for i, test in enumerate(tests):
    print(f"Test #{i+1}.")
    print("Got:")
    exec(test['input'])
    print(f"\nExpected: {test['output']}")
    print("--------" * 3)

