# Learn: Gini Impurity and Best Split in Decision Trees

## Overview

A core concept in Decision Trees (and by extension, Random Forests) is how the model chooses where to split the data at each node. One popular criterion used for splitting is **Gini Impurity**.

In this task, you will implement:
- Gini impurity computation
- Finding the best feature and threshold to split on based on impurity reduction

This helps build the foundation for how trees grow in a Random Forest.

---

## Gini Impurity

For a set of samples with class labels \( y \), the Gini Impurity is defined as:

$$
G(y) = 1 - \sum_{i=1}^{k} p_i^2
$$

Where \( p_i \) is the proportion of samples belonging to class \( i \).

A pure node (all one class) has \( G = 0 \), and higher values indicate more class diversity.

---

## Weighted Gini Impurity

Given a feature and a threshold to split the dataset into left and right subsets:

$$
G_{\text{split}} = \frac{n_{\text{left}}}{n} G(y_{\text{left}}) + \frac{n_{\text{right}}}{n} G(y_{\text{right}})
$$

We choose the split that **minimizes** $( G_{\text{split}} )$.

---

## Problem Statement

You are given a dataset $( X \in \mathbb{R}^{n \times d} )$ and labels $( y \in \{0, 1\}^n $). Implement the following functions:

### Functions to Implement

```python
def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    ...
```
