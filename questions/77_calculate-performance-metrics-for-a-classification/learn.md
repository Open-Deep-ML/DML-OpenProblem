
## Performance Metrics

Performance metrics such as accuracy, F1 score, specificity, negative predictive value, precision, and recall are vital to understanding how a model is performing.

How many observations are correctly labeled? Are we mislabeling one category more than the other? Performance metrics can answer these questions and provide an idea of where to focus to improve a model's performance.

For this problem, starting with the confusion matrix is a helpful first step, as all the elements of the confusion matrix can help with calculating other performance metrics.

For a binary classification problem of a dataset with $n$ observations, the confusion matrix is a $2 \times 2$ matrix with the following structure:

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

### Metrics

#### Accuracy
How many observations are labeled as the actual category they belong to?

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

#### Precision
How many elements labeled as positive are actually positive?

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

#### Negative Predictive Value
How many elements labeled as negative are actually negative?

$$
\text{Negative Predictive Value} = \frac{TN}{TN + FN}
$$

#### Recall
Out of all positive elements, how many were correctly labeled?

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

#### Specificity
How well are we labeling the negative elements correctly?

$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

#### F1 Score
How to account for the trade-off of false negatives and positives? The F1 score is the harmonic mean of precision and recall.

$$
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$
