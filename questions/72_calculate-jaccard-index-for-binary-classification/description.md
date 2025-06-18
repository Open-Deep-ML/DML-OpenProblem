
## Task: Implement the Jaccard Index

Your task is to implement a function `jaccard_index(y_true, y_pred)` that calculates the Jaccard Index, a measure of similarity between two binary sets. The Jaccard Index is widely used in binary classification tasks to evaluate the overlap between predicted and true labels.

### Your Task:
Implement the function `jaccard_index(y_true, y_pred)` to:
1. Calculate the Jaccard Index between the arrays `y_true` and `y_pred`.
2. Return the Jaccard Index as a float value.
3. Ensure the function handles cases where:
   - There is no overlap between `y_true` and `y_pred`.
   - Both arrays contain only zeros (edge cases).

The Jaccard Index is defined as:

$$
\scriptsize
\text{Jaccard Index} = \frac{\text{Number of elements in the intersection of } y_{\text{true}} \text{ and } y_{\text{pred}}}{\text{Number of elements in the union of } y_{\text{true}} \text{ and } y_{\text{pred}}}
$$


Where:
- $ y_{\text{true}} $ and $ y_{\text{pred}} $ are binary arrays of the same length, representing true and predicted labels.
- The result ranges from 0 (no overlap) to 1 (perfect overlap).
