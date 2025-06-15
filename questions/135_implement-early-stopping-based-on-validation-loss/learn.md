## Implementing Early Stopping Criterion

Early stopping is a regularization technique that helps prevent overfitting in machine learning models. Your task is to implement the early stopping decision logic based on the validation loss history.

### Problem Description

Given a sequence of validation losses from model training, determine if training should be stopped based on the following criteria:

- Training should stop if the validation loss hasn't improved (decreased) for a specified number of epochs (patience)
- An improvement is only counted if the loss decreases by more than a minimum threshold (min_delta)
- The best model is the one with the lowest validation loss

### Example

Consider the following validation losses: [0.9, 0.8, 0.75, 0.77, 0.76, 0.77, 0.78]

- With patience=2 and min_delta=0.01:
  - Best loss is 0.75 at epoch 2
  - No improvement > 0.01 for next 2 epochs
  - Should stop at epoch 4

### Function Requirements

- Return both the epoch to stop at and the best epoch
- If no stopping is needed, return the last epoch
- Epochs are 0-indexed
