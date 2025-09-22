### Problem

Implement the perceptron trick for logistic regression. Given training data with binary labels, update the weights using the perceptron learning rule and return the final weights and predictions.

The perceptron trick updates weights as follows:
- If prediction is correct: no update
- If prediction is wrong: $\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot y_i \cdot \mathbf{x}_i$

Where:
- $\mathbf{w}$ is the weight vector (including bias)
- $\eta$ is the learning rate
- $y_i \in \{-1, +1\}$ is the true label
- $\mathbf{x}_i$ is the feature vector (with bias term)

The prediction function is: $\hat{y} = \text{sign}(\mathbf{w}^T \mathbf{x})$

Return the final weights and predictions on the training set.
