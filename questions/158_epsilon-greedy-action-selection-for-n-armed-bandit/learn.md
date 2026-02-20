### Epsilon-Greedy Policy

The epsilon-greedy method is a fundamental action selection strategy used in reinforcement learning, especially for solving the n-armed bandit problem. The key idea is to balance **exploration** (trying new actions) and **exploitation** (choosing the best-known action):

- With probability $\varepsilon$ (epsilon), the agent explores by selecting an action at random.
- With probability $1-\varepsilon$, it exploits by choosing the action with the highest estimated value (greedy choice).

The epsilon-greedy policy is simple to implement and provides a way to avoid getting stuck with suboptimal actions due to insufficient exploration.
