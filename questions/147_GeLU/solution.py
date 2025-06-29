import numpy as np

def GeLU(x: np.ndarray) -> np.ndarray:
    return np.round(0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))),4)
