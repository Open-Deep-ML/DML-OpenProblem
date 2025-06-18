## Understanding Multi-Head Attention

Multi-head attention is a fundamental mechanism in transformer models, allowing the model to focus on different parts of the input sequence simultaneously. This enables the model to capture a wider variety of relationships and dependencies, which is crucial for handling complex data, such as natural language. By using multiple attention heads, the model learns to attend to various aspects of the input at different levels of abstraction, enhancing its ability to capture complex relationships.

### Concepts

The attention mechanism allows the model to weigh the importance of different input elements based on their relevance to a specific task. In tasks like machine translation, for example, attention helps the model focus on relevant words in a sentence to understand the overall meaning. Multi-head attention extends this concept by using multiple attention heads, each learning different representations of the input data, which improves the model’s ability to capture richer relationships and dependencies.

The process of multi-head attention involves several key steps:

1. **Computing Attention Scores:** This involves calculating how much focus each element in the input should receive based on its relationship with other elements.
2. **Applying Softmax:** The attention scores are transformed into probabilities using the softmax function, which normalizes the scores so that they sum to one.
3. **Aggregating Results:** The final output is computed by taking a weighted sum of the input values, where the weights are determined by the attention scores.

### Structure of Multi-Head Attention

The attention mechanism can be described with Query (Q), Key (K), and Value (V) matrices. The process of multi-head attention works by repeating the standard attention mechanism multiple times in parallel, with different sets of learned weight matrices for each attention head.

#### 1. Splitting Q, K, and V

Assume that the input Query (Q), Key (K), and Value (V) matrices have dimensions $(\text{seqLen}, d_{model})$, where $d_{\text{model}}$ is the model dimension. In multi-head attention, these matrices are divided into n smaller matrices, each corresponding to a different attention head. Each smaller matrix has dimensions $(\text{seqLen}, d_k)$, where $d_k = \frac{d_{\text{model}}}{n}$ is the dimensionality of each head.

For each attention $\text{head}_i$, we get its subset of Query $\text{Q}_i$, Key $\text{K}_i$, and Value $\text{V}_i$. These subsets are computed independently for each head.

#### 2. Computing Attention for Each Head

Each head independently computes its attention output. The calculation is similar to the single-head attention mechanism:

$$
\text{score}_i = \frac{Q_i K_i^T}{\sqrt{d_k}}
$$

Where $$d_k$$ is the dimensionality of the key space for each head. The scaling factor $$\frac{1}{\sqrt{d_k}}$$ ensures the dot product doesn't grow too large, preventing instability in the softmax function.

The softmax function is applied to the scores to normalize them, transforming them into attention weights for each head:

$$
\text{SoftmaxScore}_i = \text{softmax}(\text{score}_i)
$$

#### 3. Softmax Calculation and Numerical Stability

When computing the softmax function, especially in the context of attention mechanisms, there's a risk of numerical overflow or underflow, which can occur when the attention scores become very large or very small. This issue arises because the exponential function $$\exp$$ grows very quickly, and when dealing with large numbers, it can result in values that are too large for the computer to handle, leading to overflow errors.

To prevent this, we apply a common technique: subtracting the maximum score from each attention score before applying the exponential function. This helps to ensure that the largest value in the attention scores becomes zero, reducing the likelihood of overflow. Here's how it's done:

$$
\text{SoftmaxScore} = \frac{\exp(\text{score} - \text{score}_{\text{max}})}{\sum\exp(\text{score} - \text{score}_{\text{max}})}
$$

Where $$\text{score}_{i,\text{max}}$$ is the maximum value of the attention scores for the \(i\)-th head. Subtracting the maximum score from each individual score ensures that the largest value becomes 0, which prevents the exponentials from becoming too large.

This subtraction does not affect the final result of the softmax calculation because the softmax is a relative function—it's the ratios of the exponentials that matter. Therefore, this adjustment ensures numerical stability while maintaining the correctness of the computation.

To summarize, when computing softmax in multi-head attention:

- Subtract the maximum score from each attention score before applying the exponential function.
- This technique prevents overflow by ensuring that the largest value becomes 0, which keeps the exponential values within a manageable range.
- The relative relationships between the scores remain unchanged, so the softmax output remains correct.

By applying this numerical stability trick, the softmax function becomes more robust and prevents computational issues that could arise during training or inference, especially when dealing with large models or sequences.

Finally, the attention output for each $$\text{head}_i$$ is computed as:

$$
\text{head}_i = \text{SoftmaxScore}_i \cdot V_i
$$

#### 4. Concatenation and Linear Transformation

After computing the attention output for each head, the outputs are concatenated along the feature dimension. This results in a matrix of dimensions $$(\text{seqLen}, d_{\text{model}})$$, where the concatenated attention outputs are passed through a final linear transformation to obtain the final multi-head attention output.

$$
\text{MultiHeadOutput} = \text{concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_n)
$$

The concatenated result is then linearly transformed using a weight matrix $W_{\text{o}}$ to obtain the final output. However, in our case, obtaining the multi-head attention output without this final transformation is sufficient:

$$
\text{MultiHeadOutput} = W_o \cdot \text{MultiHeadOutput}
$$

### Key Points

- Each attention head processes the input independently using its own set of learned weights. This allows each head to focus on different relationships in the data.
- Each head calculates its attention scores based on its corresponding Query, Key, and Value matrices, producing different attention outputs.
- The outputs of all attention heads are concatenated to form a unified representation. This concatenated result is then linearly transformed to generate the final output.

Multi-head attention allows the model to attend to different aspects of the input sequence in parallel, making it more capable of learning complex and diverse relationships. This parallelization of attention heads enhances the model's ability to understand the data from multiple angles simultaneously, contributing to improved performance in tasks like machine translation, text generation, and more.
