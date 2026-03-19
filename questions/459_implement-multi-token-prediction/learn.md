# **Multi-Token Prediction (MTP)**

## **1. Definition**
Multi-Token Prediction is an auxiliary training objective where the model predicts not just the next token, but multiple future tokens simultaneously. Instead of a single prediction head, the model uses additional lightweight heads that each predict a different future token.

## **2. Motivation**
- **Better representations:** Predicting further into the future forces hidden states to encode more long-range information.
- **Faster inference:** MTP heads can be used for speculative decoding, where multiple token candidates are generated in one forward pass and verified in parallel.
- **Improved sample efficiency:** Each training example provides multiple supervision signals.

## **3. Architecture**
Given the main model's hidden states $H \in \mathbb{R}^{L \times d}$ and $N$ MTP heads:

For each head $k \in \{1, \ldots, N\}$:

**Step 1: Get target embeddings (shifted by k-1 positions)**
$$E_k = \text{Embed}(\text{targets}[k-1 : L-N+k-1])$$

**Step 2: Concatenate with hidden states**
$$C_k = [H_{:L-N} \; ; \; E_k] \quad \in \mathbb{R}^{(L-N) \times 2d}$$

**Step 3: Project back to hidden dimension**
$$\tilde{H}_k = C_k \cdot W_k^T \quad \in \mathbb{R}^{(L-N) \times d}$$

**Step 4: Compute logits**
$$\text{logits}_k = \tilde{H}_k \cdot W_{\text{lm}}^T \quad \in \mathbb{R}^{(L-N) \times V}$$

**Step 5: Cross-entropy loss against shifted targets**
$$\mathcal{L}_k = \text{CE}(\text{logits}_k, \text{targets}[k : L-N+k])$$

**Final loss:**
$$\mathcal{L}_{\text{MTP}} = \frac{1}{N} \sum_{k=1}^{N} \mathcal{L}_k$$

## **4. Key Details**
- Each MTP head has its own projection matrix $W_k$ but shares the embedding table and LM head with the main model
- The concatenation of hidden states with target embeddings provides "teacher forcing" — each head sees the ground truth of previous positions
- During inference, MTP heads enable speculative decoding for faster generation

## **5. Role in MiniMax M2.5**
MiniMax M2.5 uses 3 MTP modules, each containing 1 transformer layer. These auxiliary heads are trained jointly with the main model and can be used for speculative decoding during inference.
