## Task: Implement F-Score Calculation for Binary Classification

Your task is to implement a function that calculates the F-Score for a binary classification task. The F-Score combines both Precision and Recall into a single metric, providing a balanced measure of a model's performance.

Write a function `f_score(y_true, y_pred, beta)` where:

- `y_true`: A numpy array of true labels (binary).
- `y_pred`: A numpy array of predicted labels (binary).
- `beta`: A float value that adjusts the importance of Precision and Recall. When `beta=1`, it computes the F1-Score, a balanced measure of both Precision and Recall.

The function should return the F-Score rounded to three decimal places.

    
