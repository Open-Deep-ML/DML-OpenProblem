## Solution Explanation

Depthwise separable convolution is a powerful technique for building efficient neural networks. It achieves similar performance to standard convolutions while using significantly fewer parameters and computations.

### Understanding the Problem

**Standard Convolution** applies $C_{out}$ filters of size $(K \times K \times C_{in})$ to produce output with $C_{out}$ channels. The number of parameters is:

$$
\text{Params}_{standard} = K \times K \times C_{in} \times C_{out}
$$

**Depthwise Separable Convolution** splits this into two steps:

1. **Depthwise Convolution**: Applies one filter per input channel
   - Each filter has size $(K \times K \times 1)$
   - Produces $C_{in}$ output channels (one per input channel)
   - Parameters: $K \times K \times C_{in}$

2. **Pointwise Convolution**: Applies 1×1 convolution to mix channels
   - Uses $(1 \times 1 \times C_{in})$ filters to produce $C_{out}$ channels
   - Parameters: $1 \times 1 \times C_{in} \times C_{out}$

**Total Parameters**:

$$
\text{Params}_{separable} = K \times K \times C_{in} + C_{in} \times C_{out}
$$

### Parameter Reduction Factor

The reduction factor is:

$$
\frac{\text{Params}_{separable}}{\text{Params}_{standard}} = \frac{K \times K \times C_{in} + C_{in} \times C_{out}}{K \times K \times C_{in} \times C_{out}} = \frac{1}{C_{out}} + \frac{1}{K^2}
$$

For example, with $K=3$ and $C_{out}=128$:

$$
\frac{1}{128} + \frac{1}{9} \approx 0.119
$$

This means **~8.4× fewer parameters**!

### Implementation Steps

#### Step 1: Depthwise Convolution

For each input channel $c$, apply its corresponding filter independently:

$$
D_{h,w,c} = \sum_{i=0}^{K-1} \sum_{j=0}^{K-1} I_{h+i, w+j, c} \times F^{dw}_{i,j,c}
$$

Where:

- $D$ is the depthwise output
- $I$ is the input
- $F^{dw}$ is the depthwise filter
- $(h, w)$ are spatial coordinates
- $c$ is the channel index

#### Step 2: Pointwise Convolution

Apply 1×1 convolution to mix channels:

$$
O_{h,w,k} = \sum_{c=0}^{C_{in}-1} D_{h,w,c} \times F^{pw}_{c,k}
$$

Where:

- $O$ is the final output
- $F^{pw}$ is the pointwise filter (1×1 convolution weights)
- $k$ is the output channel index

### Code Implementation

```python
import numpy as np

def depthwise_separable_conv2d(
    input: np.ndarray,
    depthwise_filters: np.ndarray,
    pointwise_filters: np.ndarray
) -> np.ndarray:
    H, W, C_in = input.shape
    K, _, _ = depthwise_filters.shape
    _, _, C_out = pointwise_filters.shape

    H_out = H - K + 1
    W_out = W - K + 1

    # Step 1: Depthwise convolution
    depthwise_output = np.zeros((H_out, W_out, C_in))
    for h in range(H_out):
        for w in range(W_out):
            for c in range(C_in):
                patch = input[h:h+K, w:w+K, c]
                depthwise_output[h, w, c] = np.sum(patch * depthwise_filters[:, :, c])

    # Step 2: Pointwise convolution (1x1 conv)
    output = np.zeros((H_out, W_out, C_out))
    for h in range(H_out):
        for w in range(W_out):
            for k in range(C_out):
                output[h, w, k] = np.sum(depthwise_output[h, w, :] * pointwise_filters[0, 0, :, k])

    return output
```

### Key Insights

1. **Efficiency**: Depthwise separable convolutions are 8-9× more efficient than standard convolutions
2. **Applications**: Used in MobileNet, MobileNetV2, Xception, EfficientNet
3. **Trade-off**: Slight accuracy drop vs massive computation savings
4. **Mobile/Edge AI**: Essential for deploying models on resource-constrained devices

### Real-World Usage

```python
# MobileNetV2 uses depthwise separable convolutions extensively
# A typical block:
# 1. Pointwise expansion (1×1 conv to increase channels)
# 2. Depthwise convolution (3×3 depthwise)
# 3. Pointwise projection (1×1 conv to reduce channels)
```

### Complexity Analysis

- **Time Complexity**: $O(H_{out} \times W_{out} \times (K^2 \times C_{in} + C_{in} \times C_{out}))$
- **Space Complexity**: $O(H_{out} \times W_{out} \times C_{in})$ for intermediate depthwise output
