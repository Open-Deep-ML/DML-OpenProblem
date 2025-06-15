
## Mixture of Experts Layer

Mixture-of-Experts layers route each token through a small subset of expert networks, reducing computation while retaining flexibility.

### 1. Gating with Softmax  
- **Logits**: For each token $t$, compute a vector of gating scores $g_t \in \mathbb{R}^E$, where $E$ is the number of experts.  
- **Softmax**: Convert scores into a probability distribution  
  $$
  \alpha_{t,j}
    = \frac{\exp\bigl(g_{t,j} - \max_j g_{t,j}\bigr)}
           {\sum_{j'=1}^{E}\exp\bigl(g_{t,j'} - \max_j g_{t,j'}\bigr)}.
  $$

### 2. Top-$k$ Selection  
- **Sparsity**: Keep only the $k$ largest weights per token, zeroing out the rest.  
- **Renormalize**: For token $t$, let $\mathcal{K}_t$ be the indices of the top $k$ experts. Then  
  $$
  \tilde\alpha_{t,j} =
    \begin{cases}
      \displaystyle\frac{\alpha_{t,j}}{\sum_{i \in \mathcal{K}_t}\alpha_{t,i}}
        & j \in \mathcal{K}_t,\\[8pt]
      0
        & \text{otherwise.}
    \end{cases}
  $$

### 3. Expert Computation  
Each expert $i$ applies its own linear transform to the token embedding $x_t$:  
$$
O_t^{(i)} = x_t\,W_e^{(i)},
$$  
where $W_e^{(i)}$ is the expert's $d \times d$ weight matrix.

### 4. Weighted Aggregation  
Combine the selected experts' outputs for each token:  
$$
y_t = \sum_{i=1}^{E} \tilde\alpha_{t,i}\,O_t^{(i)}.
$$  
The result $y_t$ lives in the original embedding space $\mathbb{R}^d$.

---

### Example Walk Through

Suppose one sentence of length 2, embedding size 3, $E=4$ experts, and $k=2$.  
- After flattening, you get 2 softmax distributions of length 4.  
- You pick the top 2 experts for each token and renormalize their weights.  
- Each selected expert produces a 3-dimensional output for its tokens.  
- You weight and sum those outputs to yield the final 3-dimensional vector per token.

This sparse routing mechanism dramatically cuts computation only $k$ experts run per token instead of all $E$ while retaining the expressivity of a full ensemble.
