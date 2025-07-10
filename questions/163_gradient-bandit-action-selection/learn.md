# **Gradient Bandits**

Gradient Bandit algorithms are a family of action-selection methods for multi-armed bandit problems. Instead of estimating action values, they maintain a set of *preferences* for each action and use these to generate a probability distribution over actions via the softmax function. The algorithm then updates these preferences directly to increase the likelihood of selecting actions that yield higher rewards.

---

## **Algorithm Outline**

1. **Preferences** ($H_a$): For each action $a$, keep a real-valued preference $H_a$ (initialized to zero).
2. **Action Probabilities (Softmax):** At each timestep, choose action $a$ with probability:

$$
P(a) = \frac{e^{H_a}}{\sum_j e^{H_j}}
$$

3. **Preference Update Rule:** After receiving reward $R_t$ for selected action $A_t$, update preferences as:

$$
H_a \leftarrow H_a + \alpha \cdot (R_t - \bar{R_t}) \cdot (1 - P(a)), \text{ if } a = A_t
$$
$$
H_a \leftarrow H_a - \alpha \cdot (R_t - \bar{R_t}) \cdot P(a), \text{ if } a \neq A_t
$$
Where:
- $\bar{R_t}$ is the running average reward (baseline, helps reduce variance)
- $\alpha$ is the step size

---

## **Key Properties**
- Uses *softmax* probabilities for exploration (all actions get non-zero probability)
- Action preferences directly drive probability updates
- The baseline $\bar{R_t}$ stabilizes learning and reduces update variance
- More likely to select actions with higher expected reward

---

## **When to Use Gradient Bandits?**
- Problems where the best action changes over time (non-stationary)
- Situations requiring continuous, adaptive exploration
- Settings where value estimates are unreliable or less stable

---

## **Summary**
Gradient bandit methods offer a principled way to learn action preferences by maximizing expected reward via gradient ascent. Their use of the softmax function ensures robust, probabilistic exploration and efficient learning from feedback.
