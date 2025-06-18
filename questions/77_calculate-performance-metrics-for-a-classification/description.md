
### Task: Implement Performance Metrics Calculation

In this task, you are required to implement a function `performance_metrics(actual, predicted)` that computes various performance metrics for a binary classification problem. These metrics include:

- Confusion Matrix
- Accuracy
- F1 Score
- Specificity
- Negative Predictive Value

The function should take in two lists:

- `actual`: The actual class labels (1 for positive, 0 for negative).
- `predicted`: The predicted class labels from the model.

### Output

The function should return a tuple containing:

1. `confusion_matrix`: A 2x2 matrix.
2. `accuracy`: A float representing the accuracy of the model.
3. `f1_score`: A float representing the F1 score of the model.
4. `specificity`: A float representing the specificity of the model.
5. `negative_predictive_value`: A float representing the negative predictive value.

### Constraints

- All elements in the `actual` and `predicted` lists must be either 0 or 1.
- Both lists must have the same length.
