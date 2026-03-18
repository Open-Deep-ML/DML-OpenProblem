# **Grouped Query Attention (GQA)**

## **1. Definition**
Grouped Query Attention (GQA) is an attention variant that uses fewer key-value heads than query heads. Instead of each query head having its own dedicated KV pair (Multi-Head Attention) or all query heads sharing a single KV pair (Multi-Query Attention), GQA groups multiple query heads to share the same KV head.

## **2. Motivation**
- **Memory efficiency:** Fewer KV heads means a smaller KV cache during inference, reducing memory usage.
- **Speed:** Reduces the memory bandwidth bottleneck during autoregressive generation.
- **Quality:** Retains most of the quality of full multi-head attention while being significantly more efficient.

## **3. How It Works**
Given:
- $n_q$ query heads
- $n_{kv}$ key-value heads
- Group size $g = n_q / n_{kv}$

Each group of $g$ query heads shares the same key and value head:

$$
\text{Group}_j = \{Q_{j \cdot g}, Q_{j \cdot g + 1}, \ldots, Q_{(j+1) \cdot g - 1}\}
$$

For each query head $Q_i$ in group $j$:

$$
\text{Attention}(Q_i, K_j, V_j) = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}}\right) V_j
$$

## **4. Spectrum of Attention Variants**

| Variant | KV Heads | Description |
|---|---|---|
| Multi-Head Attention (MHA) | $n_q$ | Each Q head has its own KV |
| Grouped Query Attention (GQA) | $n_q / g$ | Groups of Q heads share KV |
| Multi-Query Attention (MQA) | 1 | All Q heads share one KV |

## **5. Role in MiniMax M2.5**
MiniMax M2.5 uses GQA with:
- 48 query heads
- 8 key-value heads
- Group size: $48 / 8 = 6$ query heads per KV group
- Head dimension: 128

This reduces the KV cache size by 6x compared to full MHA.
