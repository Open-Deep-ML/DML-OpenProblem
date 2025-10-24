import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	# Your code here
	x = np.array(X)
    w = np.array(w)
    y_true = np.array(y_true)

    y_pr = x @ w

    loss = np.mean((y_pr - y_true)**2)

    reg = alpha * np.sum(w**2)

    final_loss = loss + reg

    return final_loss
