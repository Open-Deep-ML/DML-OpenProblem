import numpy as np


def global_avg_pool(x: np.ndarray) -> np.ndarray:
    return np.mean(x, axis=(0, 1))
