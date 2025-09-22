from typing import List, Tuple
import numpy as np


def perceptron_trick(X: List[List[float]], y: List[int], learning_rate: float = 0.1, max_epochs: int = 100) -> Tuple[List[float], List[int]]:
    """
    Implement the perceptron trick for binary classification.
    
    Args:
        X: List of feature vectors (without bias term)
        y: List of binary labels (-1 or +1)
        learning_rate: Learning rate for weight updates
        max_epochs: Maximum number of training epochs
        
    Returns:
        Tuple of (final_weights, predictions)
        - final_weights: Weight vector including bias term
        - predictions: Predictions on training data
    """
    # TODO: implement
    raise NotImplementedError
