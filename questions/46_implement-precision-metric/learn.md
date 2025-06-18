
## Understanding Precision in Classification

Precision is a key metric used in the evaluation of classification models, particularly in binary classification. It provides insight into the accuracy of the positive predictions made by the model.

### Mathematical Definition
Precision is defined as the ratio of true positives (TP) to the sum of true positives and false positives (FP):
$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

Where:
- **True Positives (TP)**: The number of positive samples that are correctly identified as positive.
- **False Positives (FP)**: The number of negative samples that are incorrectly identified as positive.

### Characteristics of Precision
- **Range**: Precision ranges from 0 to 1, where 1 indicates perfect precision (no false positives) and 0 indicates no true positives.
- **Interpretation**: High precision means that the model has a low false positive rate, meaning it rarely labels negative samples as positive.
- **Use Case**: Precision is particularly useful when the cost of false positives is high, such as in medical diagnosis or fraud detection.

In this problem, you will implement a function to calculate precision given the true labels and predicted labels of a binary classification task.
