# Gridworld Policy Evaluation

Implement a function that evaluates the state-value function for a 5x5 gridworld under a given policy. In this gridworld, the agent can move in four directions: up, down, left, and right. Each move incurs a constant reward of -1, and terminal states (the four corners) remain unchanged. The policy is provided as a dictionary mapping each state (tuple: (row, col)) to a dictionary of action probabilities.

## Example

**Input:**

```python
policy = {
    (i, j): {'up': 0.25, 'down': 0.25, 'left': 0.25, 'right': 0.25}
    for i in range(5) for j in range(5)
}
gamma = 0.9
threshold = 0.001
```

## Output:

A 5x5 list of state values that converges after iterative evaluation.

```
[0.0, -4.864480919478529, -6.078955203735765, -4.864480919478529, 0.0]
[-4.864480919478529, -6.23388594292537, -6.7676569349718365, -6.233885942925371, -4.864480919478529]
[-6.078955203735764, -6.7676569349718365, -7.090189335232064, -6.7676569349718365, -6.078955203735764]
[-4.864480919478529, -6.23388594292537, -6.7676569349718365, -6.233885942925371, -4.864480919478529]
[0.0, -4.864480919478529, -6.078955203735765, -4.864480919478529, 0.0]
```

## Reasoning:

For each non-terminal state, compute the expected value over all possible actions using the policy. Update the state value iteratively using the Bellman expectation equation until the maximum change across states is below the threshold, ensuring that terminal states remain fixed.