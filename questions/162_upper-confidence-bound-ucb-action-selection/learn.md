# **Upper Confidence Bound (UCB) Action Selection**

The **Upper Confidence Bound (UCB)** is a principled method for balancing exploration and exploitation in the multi-armed bandit problem. UCB assigns each action an optimistic estimate of its potential by considering both its current estimated value and the uncertainty around that estimate.

---

## **UCB1 Formula**
Given:
- $Q(a)$: Average reward of action $a$
- $N(a)$: Number of times action $a$ has been chosen
- $t$: Total number of action selections so far
- $c$: Exploration coefficient (higher $c$ → more exploration)

The UCB value for each action $a$ is:

$$
UCB(a) = Q(a) + c \cdot \sqrt{\frac{\ln t}{N(a)}}
$$

- The first term ($Q(a)$) encourages exploitation (choose the best-known action)
- The second term encourages exploration (prefer actions tried less often)

At each timestep, select the action with the highest $UCB(a)$ value.

---

## **Key Points**
- UCB ensures every action is tried (because the exploration term is large for actions with low $N(a)$)
- As $N(a)$ increases, the uncertainty shrinks and the choice relies more on the estimated value
- $c$ tunes the trade-off: high $c$ -> more exploration, low $c$ -> more exploitation

---

## **When To Use UCB?**
- In online learning tasks (multi-armed bandits)
- In environments where you need to efficiently explore without random guessing
- In real-world scenarios: ad placement, recommendation systems, A/B testing, clinical trials

---

## **Summary**
UCB is a simple and powerful method for action selection. It works by always acting as if the best-case scenario (within a confidence bound) is true for every action, balancing learning new information and maximizing reward.
