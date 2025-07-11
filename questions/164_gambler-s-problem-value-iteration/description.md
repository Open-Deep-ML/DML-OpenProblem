A gambler has the chance to bet on a sequence of coin flips. If the coin lands heads, the gambler wins the amount staked; if tails, the gambler loses the stake. The goal is to reach 100, starting from a given capital $s$ (with $0 < s < 100$). The game ends when the gambler reaches $0$ (bankruptcy) or $100$ (goal). On each flip, the gambler can bet any integer amount from $1$ up to $\min(s, 100-s)$.

The probability of heads is $p_h$ (known). Reward is $+1$ if the gambler reaches $100$ in a transition, $0$ otherwise.

**Your Task:**
Write a function `gambler_value_iteration(ph, theta=1e-9)` that:
- Computes the optimal state-value function $V(s)$ for all $s = 1, ..., 99$ using value iteration.
- Returns the optimal policy as a mapping from state $s$ to the optimal stake $a^*$ (can return any optimal stake if there are ties).

**Inputs:**
- `ph`: probability of heads (float between 0 and 1)
- `theta`: threshold for value iteration convergence (default $1e-9$)

**Returns:**
- `V`: array/list of length 101, $V[s]$ is the value for state $s$
- `policy`: array/list of length 101, $policy[s]$ is the optimal stake in state $s$ (0 if $s=0$ or $s=100$)
