import torch

def k_fold_cross_validation(X, y, k=5, shuffle=True) -> list[tuple[list[int], list[int]]]:
    """
    Return train/test index splits for k-fold cross-validation using PyTorch.
    X: Tensor or convertible of shape (n_samples, ...)
    y: Tensor or convertible of shape (n_samples, ...)
    k: number of folds
    shuffle: whether to shuffle indices before splitting
    Returns list of (train_idx, test_idx) pairs, each as Python lists of ints.
    """
    X_t = torch.as_tensor(X)
    n_samples = X_t.size(0)
    indices = torch.arange(n_samples)
    if shuffle:
        indices = indices[torch.randperm(n_samples)]
    # compute fold sizes
    base = n_samples // k
    extras = n_samples % k
    fold_sizes = [base + (1 if i < extras else 0) for i in range(k)]
    # split into folds
    folds = []
    start = 0
    for fs in fold_sizes:
        folds.append(indices[start:start+fs].tolist())
        start += fs
    # build train/test pairs
    result = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = [idx for j, f in enumerate(folds) if j != i for idx in f]
        result.append((train_idx, test_idx))
    return result
