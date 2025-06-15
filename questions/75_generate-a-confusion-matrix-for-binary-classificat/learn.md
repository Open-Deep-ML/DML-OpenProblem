
## Generate Confusion Matrix

The confusion matrix is a very useful tool to get a better understanding of the performance of a classification model. In it, you can visualize how many data points were labeled according to their correct categories.

For a binary classification problem of a dataset with $ n $ observations, the confusion matrix is a $ 2 \times 2 $ matrix with the following structure:

$$
M = \begin{pmatrix} 
TP & FN \\
FP & TN
\end{pmatrix}
$$

Where:

- **TP**: True positives, the number of observations from the positive label that were correctly labeled as positive.
- **FN**: False negatives, the number of observations from the positive label that were incorrectly labeled as negative.
- **FP**: False positives, the number of observations from the negative label that were incorrectly labeled as positive.
- **TN**: True negatives, the number of observations from the negative label that were correctly labeled as negative.

A confusion matrix is a great starting point for computing more advanced metrics such as precision and recall that capture the model's performance.
