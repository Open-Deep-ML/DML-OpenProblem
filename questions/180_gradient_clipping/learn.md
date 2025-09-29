# **Gradient Clipping**

## **1. Definition**
Gradient clipping is a technique used in machine learning to prevent the gradients from becoming too large during training, which can destabilize the learning process. It is especially important in training deep neural networks, where gradients can sometimes explode to very large values (the "exploding gradients" problem).

**Gradient clipping** works by scaling the gradients if their norm exceeds a specified threshold (max_norm). The most common form is L2-norm clipping, where the entire gradient vector is rescaled so that its L2 norm is at most `max_norm`.

## **2. Why Use Gradient Clipping?**
* **Stabilizes Training:** Prevents the optimizer from making excessively large updates, which can cause the loss to diverge or become NaN.
* **Enables Deeper Networks:** Makes it feasible to train deeper or recurrent neural networks, where exploding gradients are more likely.
* **Improves Convergence:** Helps the model converge more reliably by keeping updates within a reasonable range.

## **3. Gradient Clipping Mechanism**
Given a gradient vector $g$ and a maximum norm $M$ (max_norm), the clipped gradient $g'$ is computed as:

$$
\text{if } \|g\|_2 \leq M: \\
\quad g' = g \\
\text{else:} \\
\quad g' = g \times \frac{M}{\|g\|_2}
$$

Where:
* $g$: The original gradient vector (numpy array)
* $M$: The maximum allowed L2 norm (max_norm)
* $\|g\|_2$: The L2 norm of $g$
* $g'$: The clipped gradient vector

**Example:**
If $g = [6, 8]$ and $M = 5$:
* $\|g\|_2 = \sqrt{6^2 + 8^2} = 10$
* Since $10 > 5$, we scale $g$ by $5/10 = 0.5$, so $g' = [3, 4]$

## **4. Applications of Gradient Clipping**
Gradient clipping is widely used in training:
* **Recurrent Neural Networks (RNNs):** To prevent exploding gradients in long sequences.
* **Deep Neural Networks:** For stable training of very deep architectures.
* **Reinforcement Learning:** Where gradients can be highly variable.
* **Any optimization problem** where gradient explosion is a risk.

Gradient clipping is a simple yet powerful tool to ensure stable and effective training in modern machine learning workflows.
