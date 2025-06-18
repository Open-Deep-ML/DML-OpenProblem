## Understanding Self-Attention

Self-attention is a core concept in modern deep learning architectures, particularly transformers. It helps a model understand relationships between elements in a sequence by comparing each element with every other element.

### Key Formula

The attention score between two elements $i$ and $j$ is calculated as:

$$
\text{Attention Score}_{i,j} = \frac{\text{Value}_i \times \text{Value}_j}{\sqrt{\text{Dimension}}}
$$

### Softmax Function

The softmax function converts raw attention scores into probabilities:

$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

### Weighted Sum

Using the softmax scores, the final value for each element is calculated as a weighted sum:

$$
\text{Final Value}_i = \sum_{j} \text{Softmax Score}_{i,j} \times \text{Value}_j
$$

### Example Calculation

Consider the following values:

- Crystal values: $[4, 2, 7, 1, 9]$
- Dimension: $1$

#### Step 1: Calculate Attention Scores

For crystal $i = 1$ ($4$):

$$
\text{Score}_{1,1} = \frac{4 \times 4}{\sqrt{1}} = 16,
\quad \text{Score}_{1,2} = \frac{4 \times 2}{\sqrt{1}} = 8,
\ldots
$$

#### Step 2: Apply Softmax

Convert scores to probabilities using softmax.

#### Step 3: Compute Weighted Sum

Multiply probabilities by crystal values and sum them to get the final value.

### Applications

Self-attention is widely used in:

- Natural Language Processing (e.g., transformers)
- Computer Vision (e.g., Vision Transformers)
- Sequence Analysis

Mastering self-attention provides a foundation for understanding advanced AI architectures.
