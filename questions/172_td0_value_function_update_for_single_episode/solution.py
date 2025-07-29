def td0_policy_evaluation(episode, V, pi, alpha):
    for (s, a, r, s_next) in episode:
        V[s] += alpha * (r + V[s_next] - V[s])