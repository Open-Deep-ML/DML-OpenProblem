## Problem

Write a Python class StepLRScheduler to implement a learning rate scheduler based on the StepLR strategy. Your class should have an __init__ method implemented to initialize with an initial_lr (float), step_size (int), and gamma (float) parameter. It should also have a **get_lr(self, epoch)** method implemented that returns the current learning rate for a given epoch (int). The learning rate should be decreased by gamma every step_size epochs. Only use standard Python. 
