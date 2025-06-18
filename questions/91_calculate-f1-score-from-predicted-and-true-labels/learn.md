## **F1 Score**

The F1 score is a widely used metric in machine learning and statistics, particularly for evaluating classification models. It is the harmonic mean of **precision** and **recall**, providing a single measure that balances the trade-off between these two metrics.

### **Key Concepts**

1. **Precision**: Precision is the fraction of true positive predictions out of all positive predictions made by the model. It measures how many of the predicted positive instances are actually correct.

    $$
    \text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}
    $$

2. **Recall**: Recall is the fraction of true positive predictions out of all actual positive instances in the dataset. It measures how many of the actual positive instances were correctly predicted.

    $$
    \text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
    $$

3. **F1 Score**: The F1 score is the harmonic mean of precision and recall, providing a balanced measure that takes both metrics into account:

    $$
    \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
    $$

### **Why Use the F1 Score?**

The F1 score is particularly useful when the dataset is imbalanced, meaning the classes are not equally represented. It provides a single metric that balances the trade-off between precision and recall, especially in scenarios where maximizing one metric might lead to a significant drop in the other.

### **Example Calculation**

Given:
y_true = [1, 0, 1, 1, 0] 


y_pred = [1, 0, 0, 1, 1] 

1. **Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)**:

    $$
    \text{TP} = 2, \quad \text{FP} = 1, \quad \text{FN} = 1
    $$

2. **Calculate Precision**:

    $$
    \text{Precision} = \frac{2}{2 + 1} = \frac{2}{3} \approx 0.667
    $$

3. **Calculate Recall**:

    $$
    \text{Recall} = \frac{2}{2 + 1} = \frac{2}{3} \approx 0.667
    $$

4. **Calculate F1 Score**:

    $$
    \text{F1 Score} = 2 \times \frac{0.667 \times 0.667}{0.667 + 0.667} = 0.667
    $$

### **Applications**

The F1 score is widely used in:
- Binary classification problems (e.g., spam detection, fraud detection).
- Multi-class classification problems (evaluated per class and averaged).
- Information retrieval tasks (e.g., search engines, recommendation systems).

Mastering the F1 score is essential for evaluating and comparing the performance of classification models.
