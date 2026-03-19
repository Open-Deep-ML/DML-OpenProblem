# **Lightning Attention (Linear Attention with Decay)**

## **1. Definition**
Lightning Attention is a linear attention mechanism that replaces the quadratic softmax attention with a recurrent formulation. It achieves $O(nd^2)$ complexity instead of $O(n^2d)$, making it efficient for very long sequences. It was developed by the MiniMax team and used in MiniMax-01.

## **2. Standard Attention vs Linear Attention**

**Standard (Softmax) Attention:**
$$O = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V \quad \in O(n^2 d)$$

**Linear Attention (kernel trick):**
$$O_t = Q_t \cdot \sum_{s \leq t} K_s^T V_s = Q_t \cdot S_t \quad \in O(n d^2)$$

Where $S_t = \sum_{s \leq t} K_s^T V_s$ is a running state that accumulates key-value outer products.

## **3. Adding Exponential Decay**
To prevent the state from growing unboundedly and to focus on more recent tokens, Lightning Attention adds an exponential decay factor $\lambda$:

$$S_t = \lambda \cdot S_{t-1} + K_t^T V_t$$
$$O_t = Q_t \cdot S_t$$

Where:
- $S_t \in \mathbb{R}^{d \times d}$ is the recurrent state
- $\lambda \in (0, 1)$ is the decay factor
- $K_t^T V_t$ is the outer product of key and value at position $t$

## **4. Recurrent Interpretation**
The decay creates an exponentially weighted sum over history:

$$S_t = \sum_{s=1}^{t} \lambda^{t-s} K_s^T V_s$$

More recent tokens contribute more strongly than distant ones, providing a natural notion of locality.

## **5. Advantages**
- **Linear complexity:** $O(nd^2)$ instead of $O(n^2d)$ — better for long sequences when $n \gg d$
- **Constant memory per step:** Only need to maintain the $d \times d$ state matrix
- **Streamable:** Can process tokens one at a time in a recurrent fashion
- **Infinite context (in theory):** No fixed context window limitation

## **6. Role in MiniMax Architecture**
In MiniMax-01, the predecessor to M2.5, Lightning Attention was used in a hybrid pattern: 7 linear attention layers followed by 1 softmax attention layer, repeated across the network. This combined the efficiency of linear attention with the expressiveness of softmax attention for long-range dependencies.
