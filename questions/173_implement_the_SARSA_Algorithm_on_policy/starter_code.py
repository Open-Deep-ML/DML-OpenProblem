def sarsa_update(transitions, initial_states, alpha, gamma, max_steps):
    """
    Perform SARSA updates on the given environment transitions.

    Args:
        transitions (dict): mapping (state, action) -> (reward, next_state)
        initial_states (list): list of starting states to simulate episodes from
        alpha (float): learning rate
        gamma (float): discount factor
        max_steps (int): maximum steps allowed per episode

    Returns:
        dict: final Q-table as a dictionary {(state, action): value}
    """
    # Your code here
    pass