## Pinball Loss (Quantile Loss)

The **Pinball Loss**, also called the **Quantile Loss**, measures how well a model predicts a specific quantile $\tau \in (0, 1)$ of the target distribution. When $\tau = 0.5$ it corresponds (up to a factor of 2) to the Mean Absolute Error, but for other quantiles it becomes an **asymmetric** loss, which is exactly what makes it useful for quantile regression.

1. **Per-sample Formula**:

   For a single observation with true value $y_i$ and prediction $\hat{y}_i$, let the error be $e_i = y_i - \hat{y}_i$. The pinball loss is:

   $$
   L_\tau(y_i, \hat{y}_i) =
   \begin{cases}
   \tau \, (y_i - \hat{y}_i) & \text{if } y_i \geq \hat{y}_i \\[4pt]
   (1 - \tau)(\hat{y}_i - y_i) & \text{if } y_i < \hat{y}_i
   \end{cases}
   $$

   A convenient closed form that avoids the branch is:

   $$
   L_\tau(y_i, \hat{y}_i) = \max\big(\tau \, e_i,\; (\tau - 1)\, e_i\big)
   $$

2. **Mean over the dataset**:

   The reported loss is the average over all $n$ observations:

   $$
   \text{PinballLoss} = \frac{1}{n} \sum_{i=1}^{n} L_\tau(y_i, \hat{y}_i)
   $$

3. **Intuition**:
   - When the model **under-predicts** ($\hat{y}_i < y_i$), the error is weighted by $\tau$.
   - When the model **over-predicts** ($\hat{y}_i > y_i$), the error is weighted by $1 - \tau$.
   - Choosing a **high** quantile (e.g. $\tau = 0.9$) makes under-prediction expensive, pushing predictions upward. Choosing a **low** quantile (e.g. $\tau = 0.1$) does the opposite.

4. **Example Calculation**:

   For the values:
   ```
   y_true = [3, -0.5, 2, 7]
   y_pred = [2.5, 0.0, 2, 8]
   quantile = 0.5
   ```

   Per-sample errors $e_i = y_i - \hat{y}_i$ are $[0.5, -0.5, 0, -1]$, giving:

   $$
   \begin{align*}
   \text{PinballLoss} &= \frac{1}{4}\big(0.5\cdot0.5 + 0.5\cdot0.5 + 0 + 0.5\cdot1\big) \\
   &= \frac{1}{4}(0.25 + 0.25 + 0 + 0.5) \\
   &= \frac{1.0}{4} = 0.25
   \end{align*}
   $$

5. **Properties**:
   - Pinball loss is always non-negative: $\text{PinballLoss} \geq 0$.
   - Perfect predictions give a loss of $0$.
   - At $\tau = 0.5$ the loss equals $\tfrac{1}{2}\,\text{MAE}$.
   - It is convex in the prediction, so it can be minimized with gradient-based optimization.
