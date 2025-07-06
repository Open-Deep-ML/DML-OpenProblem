Implement an efficient method to update the mean reward for a k-armed bandit action after receiving each new reward, **without storing the full history of rewards**. Given the previous mean estimate (Q_prev), the number of times the action has been selected (k), and a new reward (R), compute the updated mean using the incremental formula.

**Note:** Using a regular mean that stores all past rewards will eventually run out of memory. Your solution should use only the previous mean, the count, and the new reward.
