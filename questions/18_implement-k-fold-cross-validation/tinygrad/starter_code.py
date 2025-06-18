def k_fold_cross_validation_tg(X, y, k=5, shuffle=True) -> list[tuple[list[int], list[int]]]:
    """
    Return train/test index splits for k-fold cross-validation using pure Python or tinygrad.
    X: list or Tensor of shape (n_samples, ...)
    y: list or Tensor of shape (n_samples, ...)
    k: number of folds
    shuffle: whether to shuffle indices before splitting
    Returns list of (train_idx, test_idx) pairs, each as Python lists of ints.
    """
    # Your implementation here
    pass
