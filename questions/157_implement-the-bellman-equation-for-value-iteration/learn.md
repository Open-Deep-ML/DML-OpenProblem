# **The Bellman Equation**

The **Bellman equation** is a fundamental recursive equation in reinforcement learning that relates the value of a state to the values of possible next states. It provides the mathematical foundation for key RL algorithms such as value iteration and Q-learning.

---

## **Key Idea**
For each state $s$, the value $V(s)$ is the maximum expected return obtainable by choosing the best action $a$ and then following the optimal policy:

$$
V(s) = \max_{a} \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma V(s') \right]
$$

Where:
- $V(s)$: value of state $s$
- $a$: possible actions
- $P(s'|s, a)$: probability of moving to state $s'$ from $s$ via $a$
- $R(s, a, s')$: reward for this transition
- $\gamma$: discount factor ($0 \leq \gamma \leq 1$)
- $V(s')$: value of next state

---

## **How to Use**
1. **For each state:**
   - For each possible action, sum over possible next states, weighting by transition probability.
   - Add the immediate reward and the discounted value of the next state.
   - Choose the action with the highest expected value (for control).
2. **Repeat until values converge** (value iteration) or as part of other RL updates.

---

## **Applications**
- **Value Iteration** and **Policy Iteration** in Markov Decision Processes (MDP)
- **Q-learning** and other RL algorithms
- Calculating the optimal value function and policy in gridworlds, games, and general MDPs

---

## **Why It Matters**
- The Bellman equation formalizes the notion of **optimality** in sequential decision-making.
- It is a backbone for teaching agents to solve environments with rewards, uncertainty, and long-term planning.
