import numpy as np


def compute_policy_gradient(
    theta: np.ndarray, episodes: list[list[tuple[int, int, float]]]
) -> np.ndarray:
    """
    Estimate the policy gradient using REINFORCE.

    Args:
        theta: (num_states x num_actions) policy parameters.
        episodes: List of episodes, where each episode is a list of (state, action, reward).

    Returns:
        Average policy gradient (same shape as theta).
    """
    # Your code here
    pass
