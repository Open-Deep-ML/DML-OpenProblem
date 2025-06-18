## Task: Implement Recall in Binary Classification

Your task is to implement the **recall** metric in a binary classification setting. Recall is a performance measure that evaluates how effectively a machine learning model identifies positive instances from all the actual positive cases in a dataset.

You need to write a function `recall(y_true, y_pred)` that calculates the recall metric. The function should accept two inputs:

- `y_true`: A list of true binary labels (0 or 1) for the dataset.
- `y_pred`: A list of predicted binary labels (0 or 1) from the model.

Your function should return the recall value rounded to three decimal places. If the denominator (TP + FN) is zero, the recall should be 0.0 to avoid division by zero.

    
