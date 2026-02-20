import numpy as np


def simulate_markov_chain(transition_matrix, initial_state, num_steps):
    """
    Simulates a Markov Chain given a transition matrix, initial state, and number of steps.

    Parameters:
    transition_matrix : 2D numpy.ndarray, transition probabilities where each row sums to 1.
    initial_state : int, starting state index.
    num_steps : int, number of steps to simulate.

    Returns:
    numpy.ndarray, array of state indices over time, including the initial state.
    """
    states = np.zeros(num_steps + 1, dtype=int)
    states[0] = initial_state
    current_state = initial_state
    for t in range(num_steps):
        probabilities = transition_matrix[current_state]
        next_state = np.random.choice(transition_matrix.shape[1], p=probabilities)
        states[t + 1] = next_state
        current_state = next_state
    return states
