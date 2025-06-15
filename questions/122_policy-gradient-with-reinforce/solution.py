import numpy as np

def compute_policy_gradient(theta, episodes):
    def softmax(x):
        x = x - np.max(x)
        exps = np.exp(x)
        return exps / np.sum(exps)

    grad = np.zeros_like(theta)
    for episode in episodes:
        rewards = [step[2] for step in episode]
        returns = np.cumsum(rewards[::-1])[::-1]
        for t, (s, a, _), G in zip(range(len(episode)), episode, returns):
            probs = softmax(theta[s])
            grad_log_pi = np.zeros_like(theta)
            grad_log_pi[s, :] = -probs
            grad_log_pi[s, a] += 1
            grad += grad_log_pi * G
    return grad / len(episodes)
