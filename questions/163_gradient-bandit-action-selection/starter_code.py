import numpy as np

class GradientBandit:
    def __init__(self, num_actions, alpha=0.1):
        """
        num_actions (int): Number of possible actions
        alpha (float): Step size for preference updates
        """
        self.num_actions = num_actions
        self.alpha = alpha
        self.preferences = np.zeros(num_actions)
        self.avg_reward = 0.0
        self.time = 0
    def softmax(self):
        # Compute softmax probabilities from preferences
        pass
    def select_action(self):
        # Sample an action according to the softmax distribution
        pass
    def update(self, action, reward):
        # Update action preferences using the gradient ascent update
        pass
