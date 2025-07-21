def gambler_value_iteration(ph, theta=1e-9):
    # Initialize value function for states 0 to 100; terminal states 0 and 100 have value 0
    V = [0.0] * 101
    # Initialize policy array (bet amount for each state)
    policy = [0] * 101
    
    # Value iteration loop
    while True:
        delta = 0
        # Iterate over non-terminal states (1 to 99)
        for s in range(1, 100):
            # Possible actions: bet between 1 and min(s, 100 - s)
            actions = range(1, min(s, 100 - s) + 1)
            action_returns = []
            # Evaluate each action
            for a in actions:
                win_state = s + a
                lose_state = s - a
                # Reward is 1 if transition reaches 100, else 0
                reward = 1.0 if win_state == 100 else 0.0
                # Expected value: ph * (reward + V[win]) + (1 - ph) * V[lose]
                ret = ph * (reward + V[win_state]) + (1 - ph) * V[lose_state]
                action_returns.append(ret)
            # Update V[s] with the maximum expected value
            max_value = max(action_returns)
            delta = max(delta, abs(V[s] - max_value))
            V[s] = max_value
        # Check for convergence
        if delta < theta:
            break
    
    # Extract optimal policy
    for s in range(1, 100):
        actions = range(1, min(s, 100 - s) + 1)
        best_action = 0
        best_return = -float('inf')
        # Find action that maximizes expected value
        for a in actions:
            win_state = s + a
            lose_state = s - a
            reward = 1.0 if win_state == 100 else 0.0
            ret = ph * (reward + V[win_state]) + (1 - ph) * V[lose_state]
            if ret > best_return:
                best_return = ret
                best_action = a
        policy[s] = best_action
    
    return V, policy
