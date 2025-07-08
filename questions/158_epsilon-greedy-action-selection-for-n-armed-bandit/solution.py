import numpy as np

def epsilon_greedy(Q, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(len(Q))
    else:
        return int(np.argmax(Q))
