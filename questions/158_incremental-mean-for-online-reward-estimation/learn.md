### Incremental Mean Update Rule

The incremental mean formula lets you update your estimate of the mean after each new observation, **without keeping all previous rewards in memory**. For the k-th reward $R_k$ and previous estimate $Q_{k}$:

$$
Q_{k+1} = Q_k + \frac{1}{k} (R_k - Q_k)
$$

This saves memory compared to the regular mean, which requires storing all past rewards and recalculating each time. The incremental rule is crucial for online learning and large-scale problems where storing all data is impractical.
