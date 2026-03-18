# **KV Cache for Autoregressive Inference**

## **1. Definition**
A KV (Key-Value) cache stores previously computed key and value tensors during autoregressive generation, so they don't need to be recomputed at each step. This is crucial for efficient inference in transformer models.

## **2. The Problem It Solves**
During autoregressive generation (producing one token at a time):
- At step $t$, the model generates token $t$ by attending to all previous tokens $1, 2, \ldots, t-1$.
- **Without cache:** Recompute K and V for all previous tokens at every step → $O(t^2)$ total computation.
- **With cache:** Only compute K and V for the new token, append to cache → $O(t)$ per step.

## **3. How It Works**
At each generation step $t$:

1. Compute $Q_t$, $K_t$, $V_t$ for only the new token.
2. Append $K_t$ and $V_t$ to the cache:
   $$K_{\text{cache}} = [K_1; K_2; \ldots; K_t]$$
   $$V_{\text{cache}} = [V_1; V_2; \ldots; V_t]$$
3. Compute attention using $Q_t$ against the full $K_{\text{cache}}$ and $V_{\text{cache}}$:
   $$\text{output} = \text{softmax}\left(\frac{Q_t K_{\text{cache}}^T}{\sqrt{d_k}}\right) V_{\text{cache}}$$

## **4. Memory Considerations**
- Cache size grows linearly with sequence length: $O(2 \cdot n_{\text{layers}} \cdot n_{\text{kv\_heads}} \cdot L \cdot d_k)$
- Using GQA (fewer KV heads) directly reduces cache size
- MiniMax M2.5 with 8 KV heads uses 6x less cache than full 48-head MHA

## **5. Role in MiniMax M2.5**
MiniMax M2.5 uses a dynamic KV cache with:
- 8 KV heads (via GQA) to minimize cache memory
- Support for 196K context length
- FP8 quantization for reduced memory footprint
