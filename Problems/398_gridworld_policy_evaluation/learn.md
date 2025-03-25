# Gridworld Policy Evaluation

In reinforcement learning, **policy evaluation** is the process of computing the state-value function for a given policy. For a gridworld environment, this involves iteratively updating the value of each state based on the expected return following the policy.

## Key Concepts

- **State-Value Function (V):**  
  The expected return when starting from a state and following a given policy.
  
- **Policy:**  
  A mapping from states to probabilities of selecting each available action.

- **Bellman Expectation Equation:**  
  For each state \( s \):
  \[
  V(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V(s')]
  \]
  where:
  - \( \pi(a|s) \) is the probability of taking action \( a \) in state \( s \),
  - \( P(s'|s,a) \) is the probability of transitioning to state \( s' \),
  - \( R(s,a,s') \) is the reward for that transition,
  - \( \gamma \) is the discount factor.

## Algorithm Overview

1. **Initialization:**  
   Start with an initial guess (commonly zeros) for the state-value function \( V(s) \).

2. **Iterative Update:**  
   For each non-terminal state, update the state value using the Bellman expectation equation. Continue updating until the maximum change in value (delta) is less than a given threshold.

3. **Terminal States:**  
   For this example, the four corners of the grid are considered terminal, so their values remain unchanged.

This evaluation method is essential for understanding how "good" each state is under a specific policy, and it forms the basis for more advanced reinforcement learning algorithms.
