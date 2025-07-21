# **Gambler's Problem and Value Iteration**

In the Gambler's Problem, a gambler repeatedly bets on a coin flip with probability $p_h$ of heads. The goal is to reach 100 starting from some capital $s$. At each state, the gambler chooses a stake $a$ (between $1$ and $\min(s, 100-s)$). If heads, the gambler gains $a$; if tails, loses $a$. The game ends at $0$ or $100$.

The objective is to find the policy that maximizes the probability of reaching 100 (the state-value function $V(s)$ gives this probability). The value iteration update is:

$$
V(s) = \max_{a \in \text{Actions}(s)} \Big[ p_h (\text{reward} + V(s + a)) + (1-p_h)V(s-a) \Big]
$$

where the reward is $+1$ only if $s + a = 100$.

After convergence, the greedy policy chooses the stake maximizing this value. This is a classic episodic MDP, and the optimal policy may not be unique (ties are possible).
