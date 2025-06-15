Write a function that implements the Q-Learning algorithm to learn the optimal Q-table for a given Markov Decision Process (MDP). The function should take the number of states, number of actions, transition probabilities matrix, rewards matrix, list of terminal states, learning rate, discount factor, epsilon for exploration, and the number of episodes as inputs. Use these parameters to iteratively update the Q-table based on the Q-Learning update rule, employing an epsilon-greedy strategy for action selection. Ensure the function handles starting from non-terminal states and stops episodes upon reaching a terminal state.

Constraints:
- num_states: Integer greater than or equal to 1.
- num_actions: Integer greater than or equal to 1.
- P: A 3D NumPy array of shape (num_states, num_actions, num_states) where each element is a probability between 0 and 1, and each sub-array sums to 1.
- R: A 2D NumPy array of shape (num_states, num_actions) with float or integer values.
- terminal_states: A list or NumPy array of integers, each between 0 and num_states - 1, with no duplicates.
- alpha: A float between 0 and 1.
- gamma: A float between 0 and 1.
- epsilon: A float between 0 and 1.
- num_episodes: An integer greater than or equal to 1.
The function should return a 2D NumPy array of shape (num_states, num_actions) representing the learned Q-table.
