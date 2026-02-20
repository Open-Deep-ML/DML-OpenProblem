import numpy as np


def q_learning(
    num_states, num_actions, P, R, terminal_states, alpha, gamma, epsilon, num_episodes
):
    """
    Implements Q-Learning algorithm to learn the optimal Q-table for a given MDP.

    Parameters:
    - num_states: int, number of states in the MDP
    - num_actions: int, number of actions in the MDP
    - P: NumPy array of shape (num_states, num_actions, num_states), transition probabilities
    - R: NumPy array of shape (num_states, num_actions), rewards for each state-action pair
    - terminal_states: list or NumPy array of integers, indices of terminal states
    - alpha: float, learning rate
    - gamma: float, discount factor
    - epsilon: float, probability of choosing a random action in epsilon-greedy policy
    - num_episodes: int, number of episodes to train

    Returns:
    - Q: NumPy array of shape (num_states, num_actions), the learned Q-table
    """
    Q = np.zeros((num_states, num_actions))

    for episode in range(num_episodes):
        # Start from a random non-terminal state
        state = np.random.choice(
            [s for s in range(num_states) if s not in set(terminal_states)]
        )

        while state not in terminal_states:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(Q[state])

            # Sample next state based on transition probabilities
            next_state = np.random.choice(num_states, p=P[state, action])

            # Get reward
            reward = R[state, action]

            # Compute target Q-value
            if next_state in terminal_states:
                target = reward
            else:
                target = reward + gamma * np.max(Q[next_state])

            # Update Q-table
            Q[state, action] += alpha * (target - Q[state, action])

            # Transition to next state
            state = next_state

    return Q
