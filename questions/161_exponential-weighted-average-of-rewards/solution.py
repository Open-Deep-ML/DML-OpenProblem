def exp_weighted_average(Q1, rewards, alpha):
    k = len(rewards)
    value = (1 - alpha) ** k * Q1
    for i, Ri in enumerate(rewards):
        value += alpha * (1 - alpha) ** (k - i - 1) * Ri
    return value
