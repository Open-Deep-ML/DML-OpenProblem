def gridworld_policy_evaluation(policy: dict, gamma: float, threshold: float) -> list[list[float]]:
    """
    Evaluate the state-value function for a given policy on a 5x5 gridworld.
    
    Parameters:
    - policy: A dictionary mapping each state (tuple: (row, col)) to a dictionary of action probabilities.
    - gamma: Discount factor.
    - threshold: Convergence threshold.
    
    Returns:
    - A 5x5 list representing the state-value function.
    """
    grid_size = 5
    # Initialize state-value function to zeros
    V = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]
    # Define actions with their effects: up, down, left, right.
    actions = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }
    # Constant reward per move
    reward = -1
    
    while True:
        delta = 0.0
        new_V = [row[:] for row in V]
        for i in range(grid_size):
            for j in range(grid_size):
                # For simplicity, assume corners are terminal states
                if (i, j) in [(0, 0), (0, grid_size-1), (grid_size-1, 0), (grid_size-1, grid_size-1)]:
                    continue
                v = 0.0
                # Update state value based on action probabilities
                for action, prob in policy[(i, j)].items():
                    di, dj = actions[action]
                    # If the move goes off-grid, the agent stays in the same state
                    new_i = i + di if 0 <= i + di < grid_size else i
                    new_j = j + dj if 0 <= j + dj < grid_size else j
                    v += prob * (reward + gamma * V[new_i][new_j])
                new_V[i][j] = v
                delta = max(delta, abs(V[i][j] - new_V[i][j]))
        V = new_V
        if delta < threshold:
            break
    return V

def test_gridworld_policy_evaluation() -> None:
    grid_size = 5
    gamma = 0.9
    threshold = 0.001

    # Policy 1: Uniform policy for all non-terminal states.
    policy1 = {
        (i, j): {'up': 0.25, 'down': 0.25, 'left': 0.25, 'right': 0.25}
        for i in range(grid_size) for j in range(grid_size)
    }
    
    # Policy 2: Biased policy favoring 'down' and 'right'.
    policy2 = {
        (i, j): {'up': 0.1, 'down': 0.4, 'left': 0.1, 'right': 0.4}
        for i in range(grid_size) for j in range(grid_size)
    }
    
    # Policy 3: Randomized policy (for illustration, probabilities sum to 1 but vary per state)
    # Here, we provide a different fixed set for all states.
    policy3 = {
        (i, j): {'up': 0.2, 'down': 0.3, 'left': 0.3, 'right': 0.2}
        for i in range(grid_size) for j in range(grid_size)
    }
    
    policies = [policy1, policy2, policy3]
    
    for idx, policy in enumerate(policies, start=1):
        print(f"\nTesting Policy {idx}")
        # Test case 1: Verify grid dimensions
        V = gridworld_policy_evaluation(policy, gamma, threshold)
        assert len(V) == grid_size and all(len(row) == grid_size for row in V), "Grid dimension error"
        
        # Test case 2: Check that terminal states (corners) remain unchanged (value = 0)
        assert V[0][0] == 0 and V[0][grid_size-1] == 0 and V[grid_size-1][0] == 0 and V[grid_size-1][grid_size-1] == 0, "Terminal state value should be unchanged"
        
        # Test case 3: Verify that non-terminal state values are negative due to -1 reward per move
        assert V[2][2] < 0, "State value should be negative due to constant negative reward"
        print(f"All tests passed for Policy {idx}.")

if __name__ == "__main__":
    test_gridworld_policy_evaluation()