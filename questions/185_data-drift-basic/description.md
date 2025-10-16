## Problem

Implement a basic data drift check comparing two numeric datasets (reference vs. current).

Write a function `check_drift(ref, cur, mean_threshold, var_threshold)` that:

- Accepts two lists of numbers `ref` and `cur`.
- Computes the absolute difference in means and variances.
- Returns a tuple `(mean_drift, var_drift)` where each element is a boolean indicating whether drift exceeds the corresponding threshold:
	- `mean_drift = abs(mean(ref) - mean(cur)) > mean_threshold`
	- `var_drift  = abs(var(ref)  - var(cur))  > var_threshold`

Assume population variance (divide by N). Handle empty inputs by returning `(False, False)`.
