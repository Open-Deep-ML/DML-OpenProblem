# **SwiGLU Feed-Forward Network**

## **1. Definition**
The SwiGLU FFN is a gated feed-forward network that replaces the standard two-layer FFN in transformers. It uses three weight matrices instead of two, with a SiLU (Swish) activation on the gate path. This design was shown to improve transformer performance and is used in LLaMA, PaLM, MiniMax M2.5, and other modern architectures.

## **2. Standard FFN vs SwiGLU FFN**

**Standard Transformer FFN:**
$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x)$$

**SwiGLU FFN:**
$$\text{SwiGLU-FFN}(x) = W_2 \cdot (\text{SiLU}(W_1 \cdot x) \odot W_3 \cdot x)$$

Where $\odot$ denotes element-wise multiplication.

## **3. Components**

**SiLU (Sigmoid Linear Unit) activation:**
$$\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

**The three projections:**
- $W_1$ (gate projection): Projects input to intermediate space, followed by SiLU
- $W_3$ (up projection): Projects input to intermediate space (no activation)
- $W_2$ (down projection): Projects gated result back to hidden dimension

## **4. Why Three Matrices?**
The gating mechanism ($\text{SiLU}(W_1 x) \odot W_3 x$) allows the network to learn which features to pass through:
- $W_1$ computes the **gate**: which intermediate features are important
- $W_3$ computes the **value**: the actual information to pass
- The element-wise product filters the values through the gate
- $W_2$ projects back to the original dimension

## **5. Dimensions in MiniMax M2.5**
Each of the 256 experts in MiniMax M2.5 is a SwiGLU FFN:
- Input/output dimension: 3,072 (hidden_size)
- Intermediate dimension: 1,536
- $W_1$: 3,072 $\rightarrow$ 1,536 (gate)
- $W_3$: 3,072 $\rightarrow$ 1,536 (up)
- $W_2$: 1,536 $\rightarrow$ 3,072 (down)
- All projections are bias-free
