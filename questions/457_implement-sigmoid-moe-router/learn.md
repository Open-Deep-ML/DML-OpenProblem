# **Sigmoid MoE Router with Bias Correction**

## **1. Definition**
The Sigmoid MoE Router is the expert routing mechanism used in MiniMax M2.5. Unlike traditional MoE routers that use softmax scoring, this router uses **sigmoid activation** with a **learned bias correction** for expert selection, followed by weight normalization.

## **2. Why Sigmoid Instead of Softmax?**
- **Independent scoring:** Sigmoid scores each expert independently, while softmax creates competition between experts. This allows for more flexible expert utilization.
- **Better load balancing:** The learned bias correction term helps distribute tokens more evenly across experts without auxiliary losses dominating training.
- **Simpler gradient flow:** Sigmoid gradients don't depend on other experts' scores.

## **3. Algorithm**
Given hidden states $H \in \mathbb{R}^{T \times d}$, gate weights $W_g \in \mathbb{R}^{E \times d}$, and bias correction $b \in \mathbb{R}^{E}$:

**Step 1: Compute logits**
$$\text{logits} = H \cdot W_g^T \quad \in \mathbb{R}^{T \times E}$$

**Step 2: Apply sigmoid**
$$w = \sigma(\text{logits}) = \frac{1}{1 + e^{-\text{logits}}} \quad \in \mathbb{R}^{T \times E}$$

**Step 3: Bias-corrected scores for selection**
$$s = w + b \quad \in \mathbb{R}^{T \times E}$$

**Step 4: Top-k selection**
$$\text{indices} = \text{argsort}(s, \text{descending})[:, :k]$$

**Step 5: Gather actual weights (without bias)**
$$w_{\text{selected}} = \text{gather}(w, \text{indices})$$

**Step 6: Normalize**
$$\hat{w} = \frac{w_{\text{selected}}}{\sum_{j=1}^{k} w_{\text{selected}, j}}$$

## **4. Key Design Choices**
- The **bias is only used for selection**, not for the final weights. This decouples load balancing from the actual routing weights.
- The bias $b$ is a **learned parameter** that adjusts which experts are preferred, compensating for natural imbalances.
- The final weights are **normalized** so they sum to 1, making the weighted combination of expert outputs a proper convex combination.

## **5. Role in MiniMax M2.5**
- **256 experts** per layer, **top-8** selected per token
- Gate: `Linear(3072, 256, bias=False)`
- Bias correction: learned vector of size 256
- Only ~3% of experts activated per token (8/256)
