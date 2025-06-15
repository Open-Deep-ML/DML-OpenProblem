
## Understanding Recall in Classification

Recall is a metric that measures how often a machine learning model correctly identifies positive instances, also known as true positives, from all the actual positive samples in the dataset.

### Mathematical Definition

Recall, also known as sensitivity, is the fraction of relevant instances that were retrieved. It is calculated using the following equation:
$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

Where:
1. **True Positives (TP)**: The number of positive samples that are correctly identified as positive.
2. **False Negatives (FN)**: The number of positive samples that are incorrectly identified as negative.

### Task

In this problem, you will implement a function to calculate recall given the true labels and predicted labels of a binary classification task. The results should be rounded to three decimal places.
