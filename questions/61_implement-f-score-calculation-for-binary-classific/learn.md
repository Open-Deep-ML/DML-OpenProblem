
## Understanding F-Score in Classification

F-Score, also called F-measure, is a measure of predictive performance that's calculated from the Precision and Recall metrics.

### Mathematical Definition

The $F_{\beta}$ score applies additional weights, valuing one of precision or recall more than the other. When $\beta$ equals 1, also known as the **F1-Score**, it symmetrically represents both precision and recall in one metric. The F-Score can be calculated using the following formula:

$$
F_{\beta} = (1 + \beta^2) \times \frac{\text{precision} \times \text{recall}}{(\beta^2 \times \text{precision}) + \text{recall}}
$$

Where:

- **Recall**: The number of true positive results divided by the number of all samples that should have been identified as positive.
- **Precision**: The number of true positive results divided by the number of all samples predicted to be positive, including those not identified correctly.

### Implementation Instructions

In this problem, you will implement a function to calculate the **F-Score** given the true labels, predicted labels, and the Beta value of a binary classification task. The results should be rounded to three decimal places.

#### Special Case:
If the denominator is zero, the F-Score should be set to **0.0** to avoid division by zero.
