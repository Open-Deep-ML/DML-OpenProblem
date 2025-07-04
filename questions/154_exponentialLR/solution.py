class ExponentialLRScheduler:
    def __init__(self, initial_lr, gamma):
        """
        Initializes the ExponentialLR scheduler.

        Args:
            initial_lr (float): The initial learning rate.
            gamma (float): The multiplicative factor of learning rate decay per epoch.
                           (e.g., 0.9 for reducing LR by 10% each epoch).
        """
        self.initial_lr = initial_lr
        self.gamma = gamma

    def get_lr(self, epoch):
        """
        Calculates and returns the current learning rate for a given epoch,
        rounded to 4 decimal places.

        Args:
            epoch (int): The current epoch number (0-indexed).

        Returns:
            float: The calculated learning rate for the current epoch, rounded to 4 decimal places.
        """
        # Apply the decay factor 'gamma' raised to the power of the current 'epoch'
        # to the initial learning rate.
        current_lr = self.initial_lr * (self.gamma ** epoch)
        
        # Round the learning rate to 4 decimal places
        return round(current_lr, 4)