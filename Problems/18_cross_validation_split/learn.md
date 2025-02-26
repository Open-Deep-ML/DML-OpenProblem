## Understanding K-Fold Cross-Validation

K-Fold Cross-Validation is a resampling technique used to evaluate machine learning models by partitioning the dataset into multiple folds.

### How it Works
1. The dataset is split into **k** equal (or almost equal) parts called folds.
2. Each fold is used **once** as a test set, while the remaining **k-1** folds form the training set.
3. The process is repeated **k times**, ensuring each fold serves as a test set exactly once.

### Why Use K-Fold Cross-Validation?
- It provides a more **robust** estimate of model performance than a single train-test split.
- Reduces bias introduced by a single training/testing split.
- Allows evaluation across multiple data distributions.

### Implementation Steps
1. Shuffle the data if required.
2. Split the dataset into **k** equal (or nearly equal) folds.
3. Iterate over each fold, using it as the test set while using the remaining data as the training set.
4. Return train-test indices for each iteration.

By implementing this function, you will learn how to **split a dataset for cross-validation**, a crucial step in model evaluation.