
## Understanding Random Subsets of a Dataset

Generating random subsets of a dataset is a useful technique in machine learning, particularly in ensemble methods like bagging and random forests. By creating random subsets, models can be trained on different parts of the data, which helps in reducing overfitting and improving generalization.

### Problem Overview
In this problem, you will write a function to generate random subsets of a given dataset. Specifically:
- Given a 2D numpy array $X$, a 1D numpy array $y$, an integer `n_subsets`, and a boolean `replacements`, the function will create a list of `n_subsets` random subsets.
- Each subset will be a tuple of $(X_{\text{subset}}, y_{\text{subset}})$.

### Parameters
- **$X$**: A 2D numpy array representing the features.
- **$y$**: A 1D numpy array representing the labels.
- **$n_{\text{subsets}}$**: The number of random subsets to generate.
- **`replacements`**: A boolean indicating whether to sample with or without replacement.
  - If `replacements` is **True**, subsets will be created *with* replacements, meaning samples can be repeated within a subset.
  - If `replacements` is **False**, subsets will be created *without* replacements, meaning samples cannot be repeated within a subset.

### Importance
By understanding and implementing this technique, you can enhance the performance of your models through methods like bootstrapping and ensemble learning.
