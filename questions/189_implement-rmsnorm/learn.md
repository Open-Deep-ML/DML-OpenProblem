# **RMSNorm (Root Mean Square Layer Normalization)**

## **1. Definition**
RMSNorm is a simplified variant of Layer Normalization that normalizes activations using only the root mean square (RMS) statistic, without computing or subtracting the mean. It was introduced as a more efficient alternative to LayerNorm and is used in modern architectures like LLaMA, Gemma, and MiniMax M2.5.

## **2. Why RMSNorm over LayerNorm?**
Standard Layer Normalization computes both the mean and variance of activations, then re-centers and re-scales. RMSNorm skips the mean centering step:
- **Faster computation:** One fewer reduction operation per forward pass.
- **Comparable performance:** Empirically achieves similar or better results on large language models.
- **Widely adopted:** Used in most modern transformer architectures (LLaMA, Mistral, MiniMax M2.5, etc.).

## **3. Formula**

Given an input vector $x \in \mathbb{R}^d$:

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}
$$

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma
$$

Where:
- $d$ is the feature dimension
- $\epsilon$ is a small constant for numerical stability (e.g., $10^{-6}$)
- $\gamma$ is a learnable scale parameter (weight vector) of the same dimension as $x$

## **4. Comparison with LayerNorm**

| | LayerNorm | RMSNorm |
|---|---|---|
| Mean subtraction | Yes | No |
| Variance normalization | Yes | Yes (via RMS) |
| Learnable parameters | $\gamma$ (scale) + $\beta$ (shift) | $\gamma$ (scale) only |
| Speed | Slower | Faster |

## **5. Role in MiniMax M2.5**
In the MiniMax M2.5 architecture, RMSNorm (with $\epsilon = 10^{-6}$) is applied:
- Before self-attention (input layernorm)
- Before the MoE block (post-attention layernorm)
- After all decoder layers (final norm)
- On Q and K projections (QK normalization)
