import math

class CosineAnnealingLRScheduler:
    def __init__(self, initial_lr, T_max, min_lr):
        """
        Initializes the CosineAnnealingLR scheduler.

        Args:
            initial_lr (float): The initial (maximum) learning rate.
            T_max (int): The maximum number of epochs in the cosine annealing cycle.
                         The learning rate will reach min_lr at this epoch.
            min_lr (float): The minimum learning rate.
        """
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.min_lr = min_lr

    def get_lr(self, epoch):
        """
        Calculates and returns the current learning rate for a given epoch,
        following a cosine annealing schedule and rounded to 4 decimal places.

        Args:
            epoch (int): The current epoch number (0-indexed).

        Returns:
            float: The calculated learning rate for the current epoch, rounded to 4 decimal places.
        """
        # Ensure epoch does not exceed T_max for the calculation cycle,
        # as the cosine formula is typically defined for e from 0 to T_max.
        # Although in practice, schedulers might restart or hold LR after T_max.
        # For this problem, we'll clamp it to T_max if it goes over.
        current_epoch = min(epoch, self.T_max)

        # Calculate the learning rate using the Cosine Annealing formula
        # LR_e = LR_min + 0.5 * (LR_initial - LR_min) * (1 + cos(e / T_max * pi))
        lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
             (1 + math.cos(current_epoch / self.T_max * math.pi))
        
        # Round the learning rate to 4 decimal places
        return round(lr, 4)