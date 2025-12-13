\## NT-Xent Loss for Self-Supervised Contrastive Learning



In self-supervised contrastive learning frameworks like \*\*SimCLR\*\*, we learn meaningful representations without labels by:

1\. Creating two augmented "views" of each image

2\. Training the model to recognize that views of the \*\*same\*\* image should have similar embeddings

3\. While views of \*\*different\*\* images should have dissimilar embeddings



\### The Problem

You are given a batch of $N$ images. For each image, we generate 2 augmented views, resulting in a batch size of $2N$. The embeddings are organized in an \*\*interleaved\*\* fashion:

\- Rows $2k$ and $2k+1$ are two views of the same image $k$ (a positive pair).

\- Any other pair of rows constitutes a negative pair.



For a specific sample $i$, let $j$ be its positive pair. The \*\*NT-Xent (Normalized Temperature-scaled Cross-Entropy)\*\* loss for sample $i$ is defined as:



$$

\\ell\_i = -\\log \\frac{\\exp(\\text{sim}(z\_i, z\_j) / \\tau)}{\\sum\_{k=1}^{2N} \\mathbb{1}\_{\[k \\neq i]} \\exp(\\text{sim}(z\_i, z\_k) / \\tau)}

$$



Where:

\- $z$ is the batch of L2-normalized embeddings.

\- $\\text{sim}(u, v) = u^\\top v$ (Cosine similarity, since $u, v$ are normalized).

\- $\\mathbb{1}\_{\[k \\neq i]}$ is an indicator function (returns 1 if $k \\neq i$, else 0). Effectively, we sum over all samples except the sample itself.

\- $\\tau$ is the temperature parameter.



The total loss is the arithmetic mean over all $2N$ samples: $L = \\frac{1}{2N} \\sum\_{i=0}^{2N-1} \\ell\_i$.



\### Your Task

Implement the function `nt\_xent\_loss(z, temperature)` that computes the NT-Xent loss using vectorized NumPy operations.



\*\*Input Format\*\*

\- `z`: A numpy array of shape `(2N, embedding\_dim)` containing \*\*L2-normalized\*\* embeddings.

&nbsp; - \*\*Structure\*\*: The rows are interleaved such that `z\[2k]` and `z\[2k+1]` form a positive pair (two views of image $k$).

&nbsp; - Visually: `\[View1\_Img1, View2\_Img1, View1\_Img2, View2\_Img2, ...]`.

&nbsp; - All other interactions `z\[i]` and `z\[j]` (where `j` is not the pair of `i`) are considered negatives.

\- `temperature`: A float scaling parameter ($\\tau > 0$).



\### Output Format

\- Returns `float`: The average NT-Xent loss over all $2N$ samples.



\### Note on Stability

\- You should implement the \*\*Log-Sum-Exp trick\*\* (subtracting the maximum value before exponentiation) to ensure numerical stability.



\### Constraints

\- $N \\geq 1$ (at least 1 image, so batch size $\\geq 2$)

\- `embedding\_dim` $\\geq 1$

\- `temperature` $> 0$

\- Input embeddings are guaranteed to be L2-normalized

\- \*\*Performance:\*\* Avoid explicit `for` loops. Use matrix operations and broadcasting.

