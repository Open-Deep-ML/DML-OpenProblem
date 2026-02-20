def incremental_mean(Q_prev, k, R):
    return Q_prev + (1 / k) * (R - Q_prev)
