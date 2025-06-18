
## Task: Compute the Dice Score

Your task is to implement a function `dice_score(y_true, y_pred)` that calculates the Dice Score, also known as the Sørensen-Dice coefficient or F1-score, for binary classification. The Dice Score is used to measure the similarity between two sets and is particularly useful in tasks like image segmentation and binary classification.

### Your Task:
Implement the function `dice_score(y_true, y_pred)` to:
1. Calculate the Dice Score between the arrays `y_true` and `y_pred`.
2. Return the Dice Score as a float value rounded to 3 decimal places.
3. Handle edge cases appropriately, such as when there are no true or predicted positives.

The Dice Score is defined as:

$$
\scriptsize
\text{Dice Score} =
\frac{2 \times (\text{Number of elements in the intersection of } y_{\text{true}} \text{ and } y_{\text{pred}})}{\text{Number of elements in } y_{\text{true}} + \text{Number of elements in } y_{\text{pred}}}
$$

Where:
- $ y_{\text{true}} $ and $ y_{\text{pred}} $ are binary arrays of the same length, representing true and predicted labels.
- The result ranges from 0 (no overlap) to 1 (perfect overlap).
