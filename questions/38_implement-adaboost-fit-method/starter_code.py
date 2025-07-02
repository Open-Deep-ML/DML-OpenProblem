import numpy as np


def adaboost_fit(X, y, n_clf):
    n_samples, n_features = np.shape(X)
    np.full(n_samples, (1 / n_samples))
    clfs = []

    # Your code here

    return clfs
