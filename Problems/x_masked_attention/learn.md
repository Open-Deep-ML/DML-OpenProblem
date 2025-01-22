## Understanding Masked Attention

Masked attention is a variation of the attention mechanism used primarily in sequence modeling tasks, such as language modeling and text generation. The key idea behind masked attention is to control the flow of information by selectively masking certain elements in the input sequence. This ensures that the model attends only to valid positions when computing attention scores.

Masked attention is particularly useful in autoregressive tasks where future information should not influence the current prediction. By masking out future tokens, the model is constrained to attend only to preceding tokens or the current token, preserving causality during training and inference.

### Concepts

The attention mechanism enables the model to weigh the importance of different elements in the input sequence based on their relevance to a specific task. Masked attention modifies this process by incorporating a mask, which defines which elements the model is allowed to attend to. This ensures that the attention mechanism respects temporal or structural constraints, such as the directionality of time in sequence data.

The process of masked attention involves the following steps:

1. **Computing Attention Scores:** The model calculates how much focus each element in the sequence should receive based on its relationship with other elements.
2. **Applying the Mask:** A mask is applied to restrict attention to specific positions in the sequence. Elements outside the allowed range are effectively ignored.
3. **Normalizing Scores:** The masked scores are transformed into probabilities using the softmax function.
4. **Computing the Output:** The final output is computed as a weighted sum of the input values, with weights determined by the normalized attention scores.

### Structure of Masked Attention

The attention mechanism can be described using Query (Q), Key (K), and Value (V) matrices. In masked attention, these matrices interact with an additional mask to determine the attention distribution.

#### 1. Query, Key, and Value Matrices

- **Query (Q):** Represents the current element for which the model is computing attention.
- **Key (K):** Encodes information about all elements in the sequence.
- **Value (V):** Contains the representations that will be aggregated into the output.

Assume that the input sequence has a length of $\text{seqLen}$ and the model dimension is $d_{\text{model}}$. The dimensions of the Q, K, and V matrices are:

- Query (Q): $(\text{seqLen}, d_{\text{model}})$
- Key (K): $(\text{seqLen}, d_{\text{model}})$
- Value (V): $(\text{seqLen}, d_{\text{model}})$

#### 2. Computing Attention Scores

The raw attention scores are computed as the scaled dot product between the Query (Q) and Key (K) matrices:

$$
\text{score} = \frac{QK^T}{\sqrt{d_k}}
$$

Where $d_k$ is the dimensionality of the key space. The scaling factor $\frac{1}{\sqrt{d_k}}$ ensures that the dot product values do not grow excessively large, preventing instability in the softmax function.

#### 3. Applying the Mask

The mask is used to control which elements the model is allowed to attend to. Typically, the mask is a binary matrix of dimensions $(\text{seqLen}, \text{seqLen})$, where:

- A value of 0 indicates that attention is allowed.
- A value of $-\infty$ (or a very large negative value) indicates that attention is prohibited.

The raw attention scores are modified by adding the mask:

$$
\text{maskedScore} = \text{score} + \text{mask}
$$

This ensures that prohibited positions receive attention scores that are effectively $-\infty$, making their softmax probabilities zero.

#### 4. Softmax Calculation

The softmax function is applied to the masked scores to compute attention weights. To ensure numerical stability, the maximum score in each row is subtracted before applying the softmax function:

$$
\text{SoftmaxScore} = \frac{\exp(\text{maskedScore} - \text{maskedScore}_{\text{max}})}{\sum\exp(\text{maskedScore} - \text{maskedScore}_{\text{max}})}
$$

#### 5. Computing the Output

The final attention output is computed as a weighted sum of the Value (V) matrix, with weights determined by the attention scores:

$$
\text{output} = \text{SoftmaxScore} \cdot V
$$

### Key Points

- **Masking Future Tokens:** In autoregressive tasks, a triangular mask is used to prevent the model from attending to future positions. For a sequence of length $n$, the mask is an upper triangular matrix with 0s in the lower triangle and $-\infty$ in the upper triangle.

  Example:
  $$
  \text{mask} = \begin{bmatrix}
  0 & -\infty & -\infty \\
  0 & 0 & -\infty \\
  0 & 0 & 0
  \end{bmatrix}
  $$

- **Numerical Stability:** Subtracting the maximum score before applying softmax ensures numerical stability and prevents overflow or underflow errors.
- **Flexibility:** The mask can be customized to handle other constraints, such as ignoring padding tokens in variable-length sequences.

By selectively controlling the flow of information through masking, masked attention ensures that the model respects temporal or structural constraints, enabling it to generate coherent and contextually accurate outputs in sequence modeling tasks.
