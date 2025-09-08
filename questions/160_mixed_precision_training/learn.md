# **Mixed Precision Training**
## **1. Definition**
Mixed Precision Training is a **deep learning optimization technique** that uses both **float16** (half precision) and **float32** (single precision) data types during training to reduce memory usage and increase training speed while maintaining model accuracy.
The technique works by:
- **Using float16 for forward pass computations** to save memory and increase speed
- **Using float32 for gradient accumulation** to maintain numerical precision
- **Applying loss scaling** to prevent gradient underflow in float16
---
## **2. Key Components**
### **Mean Squared Error (MSE) Loss**
The loss function must be computed as Mean Squared Error:
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$
where $y_i$ is the target and $\hat{y}_i$ is the prediction for sample $i$.

### **Loss Scaling**
To prevent gradient underflow in float16, gradients are scaled up during the forward pass:
$$
\text{scaled\_loss} = \text{MSE} \times \text{scale\_factor}
$$
Then unscaled during backward pass:
$$
\text{gradient} = \frac{\text{scaled\_gradient}}{\text{scale\_factor}}
$$
### **Overflow Detection**
Check for invalid gradients (NaN or Inf) that indicate numerical overflow:
$$
\text{overflow} = \text{any}(\text{isnan}(\text{gradients}) \text{ or } \text{isinf}(\text{gradients}))
$$
---
## **3. Precision Usage**
- **float16**: Forward pass computations, activations, temporary calculations
- **float32**: Gradient accumulation, parameter updates, loss scaling
- **Automatic casting**: Convert between precisions as needed
- **Loss computation**: Use MSE as the loss function before scaling
---
## **4. Benefits and Applications**
- **Memory Efficiency**: Reduces memory usage by ~50% for activations
- **Speed Improvement**: Faster computation on modern GPUs with Tensor Cores
- **Training Stability**: Loss scaling prevents gradient underflow
- **Model Accuracy**: Maintains comparable accuracy to full precision training
Common in training large neural networks where memory is a constraint and speed is critical.
---