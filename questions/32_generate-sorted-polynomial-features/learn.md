## Understanding Polynomial Features

Generating polynomial features is a method used to create new features for a machine-learning model by raising existing features to a specified power. This technique helps capture non-linear relationships between features.

### Example
Given a dataset with two features $x_1$ and $x_2$, generating polynomial features up to degree 2 will create new features such as:
- $x_1^2$
- $x_2^2$
- $x_1 x_2$

### Problem Overview
In this problem you will write a function to **generate** polynomial features **and then sort each sample's features in ascending order**. Specifically:
- Given a 2-D NumPy array **X** and an integer **degree**, create a new 2-D array with all polynomial combinations of the features up to the specified degree.
- Finally, sort each row from the lowest value to the highest value.

### Importance
Polynomial expansion allows otherwise linear models to handle non-linear data. Sorting the expanded features can be useful for certain downstream tasks (e.g., histogram-based models or feature selection heuristics) and reinforces array-manipulation skills in NumPy.
