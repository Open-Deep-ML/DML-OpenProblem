## Multi-class Cross-Entropy Loss Implementation

Cross-entropy loss, also known as log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. For multi-class classification tasks, we use the categorical cross-entropy loss.

### Mathematical Background

For a single sample with C classes, the categorical cross-entropy loss is defined as:

$L = -\sum_{c=1}^{C} y_c \log(p_c)$

where:

- $y_c$ is a binary indicator (0 or 1) if class label c is the correct classification for the sample
- $p_c$ is the predicted probability that the sample belongs to class c
- $C$ is the number of classes

### Implementation Requirements

Your task is to implement a function that computes the average cross-entropy loss across multiple samples:

$L_{batch} = -\frac{1}{N}\sum_{n=1}^{N}\sum_{c=1}^{C} y_{n,c} \log(p_{n,c})$

where N is the number of samples in the batch.

### Important Considerations

- Handle numerical stability by adding a small epsilon to avoid log(0)
- Ensure predicted probabilities sum to 1 for each sample
- Return average loss across all samples
- Handle invalid inputs appropriately

The function should take predicted probabilities and true labels as input and return the average cross-entropy loss.
