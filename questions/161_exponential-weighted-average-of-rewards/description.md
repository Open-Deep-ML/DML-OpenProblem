Given an initial value $Q_1$, a list of $k$ observed rewards $R_1, R_2, \ldots, R_k$, and a step size $\alpha$, implement a function to compute the exponentially weighted average as:

$$(1-\alpha)^k Q_1 + \sum_{i=1}^k \alpha (1-\alpha)^{k-i} R_i$$

This weighting gives more importance to recent rewards, while the influence of the initial estimate $Q_1$ decays over time. Do **not** use running/incremental updates; instead, compute directly from the formula. (This is called the *exponential recency-weighted average*.)
