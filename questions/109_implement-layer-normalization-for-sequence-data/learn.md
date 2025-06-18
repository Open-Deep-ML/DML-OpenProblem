## Understanding Layer Normalization

Layer Normalization (LN) is a technique commonly used in natural language processing (NLP) tasks to normalize the activations within each individual sample (or instance) across all the features (or dimensions). Unlike **Batch Normalization**, which normalizes across the batch dimension, Layer Normalization normalizes across the feature (or channel) dimension. This makes it particularly useful for sequence-based tasks, where the sequence length varies or where the batch size is small, as is often the case in NLP applications.

### Concepts

Layer Normalization operates on the principle of normalizing each input sample independently, across its feature dimensions. This ensures that all the features in a single instance (or sequence) are scaled and shifted to have similar statistics, which helps stabilize the training process and improve the model's ability to learn.

The process of Layer Normalization consists of the following steps:

1. **Compute the Mean and Variance for Each Sample:** For each input sequence, compute the mean and variance across the feature dimensions.
2. **Normalize the Inputs:** Normalize the activations by subtracting the mean and dividing by the standard deviation.
3. **Apply Scale and Shift:** After normalization, apply a learned scale (gamma) and shift (beta) to allow the model to restore the original distribution of the data, if necessary.

### Structure of Layer Normalization for (Batch Size, Sequence Length, Feature Dimension) Input

For an input tensor with the shape **(batch_size, seq_len, d_model)** (where:
- **batch_size**: number of sequences in a batch,
- **seq_len**: sequence length (number of tokens or time steps),
- **d_model**: number of features (model dimension)),
Layer Normalization operates over the **d_model** (features) dimension, normalizing each sequence independently.

#### 1. Mean and Variance Calculation for Each Sample

- For each individual sequence in the batch (for each **b** in **batch_size**), the **mean** $\mu_b$ and **variance** $\sigma_b^2$ are computed over the features (or channels) for that particular sequence. Importantly, this computation does not involve the batch dimension, meaning each sequence is normalized independently.

  $$ 
  \mu_b = \frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} x_{b,i}
  $$

  $$
  \sigma_b^2 = \frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} (x_{b,i} - \mu_b)^2
  $$

  Where:
  - $x_{b,i}$ is the activation at batch index $b$ and feature index $i$ (across the sequence length).
  - $d_{\text{model}}$ is the model dimension (the number of features).

#### 2. Normalization

Once the mean $\mu_b$ and variance $\sigma_b^2$ have been computed for each sequence, the next step is to **normalize** the input by subtracting the mean and dividing by the standard deviation (square root of variance plus a small constant $\epsilon$ for numerical stability):

$$
\hat{x}_{b,i} = \frac{x_{b,i} - \mu_b}{\sqrt{\sigma_b^2 + \epsilon}}
$$

Where:
- $\hat{x}_{b,i}$ is the normalized value of the $i$-th feature for the $b$-th sequence.
- $\epsilon$ is a small constant added to the variance for numerical stability.

#### 3. Scale and Shift

After normalization, the next step is to apply a **scale** ($\gamma$) and **shift** ($\beta$) to the normalized activations for each sequence. These are learned parameters that allow the model to adjust the output distribution for each sequence:

$$
y_{b,i} = \gamma \hat{x}_{b,i} + \beta
$$

Where:
- $\gamma$ is the scaling factor for the feature $i$.
- $\beta$ is the shifting factor for the feature $i$.

#### 4. Training and Inference

- **During Training**: For each sequence, the mean and variance are computed over the feature dimensions and used for normalization.
- **During Inference**: The model uses the running averages of the statistics (mean and variance) computed during training to ensure consistent behavior during inference.

### Why Use Layer Normalization in NLP?

Layer Normalization is especially useful in NLP and sequence-based tasks because of the following reasons:

- **Independence from Batch Size**: Layer Normalization operates independently for each sample (sequence), which means it does not depend on the batch size. This is important in NLP tasks where the batch size can vary, or be small, which would make Batch Normalization less effective.

- **Variable Sequence Lengths**: NLP models often work with sequences of varying lengths. Layer Normalization normalizes over the feature dimension, making it easier to handle sequences of different lengths without the need for special adjustments.

- **Training Stability**: Layer Normalization helps stabilize the training process by ensuring that the activations within each sequence are normalized, preventing the network from becoming sensitive to the scale of the inputs and improving gradient flow.

### Why Not Use Batch Normalization in NLP?

Batch Normalization (BN) normalizes over the batch dimension, which works well when the batch size is large and fixed. However, in NLP tasks, there are a few reasons why **Batch Normalization** is less commonly used:

- **Batch Size Variability**: In NLP, the batch size can vary across training and inference steps. A smaller or variable batch size can lead to poor estimates of the mean and variance during Batch Normalization, which can degrade performance.

- **Sequence Length Variability**: In NLP, the length of input sequences can vary greatly (e.g., sentences of different lengths). Batch Normalization requires that the statistics be computed over the batch, which makes it difficult to apply across sequences of varying lengths without padding or truncation.

- **Dependence on Batch Statistics**: Since Batch Normalization relies on batch statistics, it can cause issues when used in tasks with smaller or highly variable batch sizes, such as in NLP, where each sequence may not represent a meaningful distribution of activations across the batch.

### Key Points

- **Normalization Over Features**: Layer Normalization normalizes across the feature dimensions (model dimension), rather than across the batch dimension, making it ideal for NLP tasks where the batch size may vary or be small.

- **Sequence-Based Normalization**: By normalizing each sequence independently, Layer Normalization ensures that the activations within a single sequence are normalized, without needing information from other sequences in the batch.

- **Stabilizing Training**: Layer Normalization improves the gradient flow and ensures that activations within each sequence are consistent, which stabilizes training and helps prevent vanishing or exploding gradients.

- **Better for Small or Variable Batch Sizes**: Layer Normalization works well with smaller batch sizes, which are often used in NLP tasks like language modeling, machine translation, and text classification.

### Summary

Layer Normalization is particularly effective in NLP tasks because it normalizes each sequence independently, ensuring that each sample has a consistent activation distribution. It is preferable over Batch Normalization in cases where the batch size is small or variable, and when sequences have different lengths, making it a popular choice for sequence-based models like transformers, BERT, and GPT.
