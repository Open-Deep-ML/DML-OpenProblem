### Exponential Recency-Weighted Average

When the environment is nonstationary, it is better to give more weight to recent rewards. The formula $$(1-\alpha)^k Q_1 + \sum_{i=1}^k \alpha (1-\alpha)^{k-i} R_i$$ computes the expected value by exponentially decaying the influence of old rewards and the initial estimate. The parameter $\alpha$ controls how quickly old information is forgotten: higher $\alpha$ gives more weight to new rewards.
