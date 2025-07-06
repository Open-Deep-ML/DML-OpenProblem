import numpy as np

def bellman_update(V, transitions, gamma):
    n_states = len(V)
    new_V = np.zeros_like(V)
    for s in range(n_states):
        action_values = []
        for a in transitions[s]:
            total = 0
            for prob, next_s, reward, done in transitions[s][a]:
                total += prob * (reward + gamma * (0 if done else V[next_s]))
            action_values.append(total)
        new_V[s] = max(action_values)
    return new_V
