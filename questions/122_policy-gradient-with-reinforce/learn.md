## REINFORCE and Policy Gradient Estimation

The REINFORCE algorithm computes gradients of the expected return with respect to policy parameters using Monte Carlo samples of episodes.

### Softmax Policy
Given $\theta$ with shape (num_states, num_actions), we define the probability of action $a$ in state $s$ as:

$$
\pi(a \mid s; \theta) = \frac{\exp(\theta[s, a])}{\sum_{a'} \exp(\theta[s, a'])}
$$

### REINFORCE Gradient
For an episode with transitions $(s_t, a_t, r_t)$ and returns $G_t = \sum_{k=t}^T r_k$:

$$
\nabla_\theta J(\theta) \approx \sum_t \nabla_\theta \log \pi(a_t \mid s_t) \cdot G_t
$$

### Log-Policy Gradient
To compute $\nabla_\theta \log \pi(a_t \mid s_t)$:

- For $\theta[s_t, a_t]$: $1 - \pi(a_t \mid s_t)$
- For $\theta[s_t, a']$, where $a' \neq a_t$: $-\pi(a' \mid s_t)$
- All other entries: 0

### Final Estimate
For multiple episodes:

$$
\hat{\nabla}_\theta J(\theta) = \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T_i} \nabla_\theta \log \pi(a_t^i \mid s_t^i) \cdot G_t^i
$$

This algorithm works even without value function estimation, making it a foundational method in policy-based reinforcement learning.
