Write a Python function to determine whether a machine learning model is overfitting, underfitting, or performing well based on training and test accuracy values. The function should take two inputs: `training_accuracy` and `test_accuracy`. It should return one of three values: 1 if Overfitting, -1 if Underfitting, or 0 if a Good fit. The rules for determination are as follows: 
- **Overfitting**: The training accuracy is significantly higher than the test accuracy (difference > 0.2).
- **Underfitting**: Both training and test accuracy are below 0.7.
- **Good fit**: Neither of the above conditions is true.
