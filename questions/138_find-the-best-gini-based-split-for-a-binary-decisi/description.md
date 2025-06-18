Implement a function that scans every feature and threshold in a small data set, then returns the split that minimises the weighted Gini impurity. Your implementation should support binary class labels (0 or 1) and handle ties gracefully.  

You will write **one** function:

```python
find_best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float]
```

* **`X`** is an $n\times d$ NumPy array of numeric features.
* **`y`** is a length-$n$ NumPy array of 0/1 labels.
* The function returns `(best_feature_index, best_threshold)` for the split with the **lowest** weighted Gini impurity.
* If several splits share the same impurity, return the first that you encounter while scanning features and thresholds.
