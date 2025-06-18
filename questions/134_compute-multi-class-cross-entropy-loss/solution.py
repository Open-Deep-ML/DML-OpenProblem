import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray,epsilon = 1e-15) -> float:

    predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)

    #Write your code here
    log_probs = np.log(predicted_probs)
    loss = -np.sum(true_labels * log_probs, axis=1)
    return float(np.mean(loss))
