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
    if not X or not y:
        return [], []
    
    n_features = len(X[0])
    n_samples = len(X)
    
    # Initialize weights (including bias term)
    weights = [0.0] * (n_features + 1)
    
    # Add bias term to each feature vector
    X_with_bias = []
    for x in X:
        X_with_bias.append(x + [1.0])  # Add bias term
    
    # Convert to numpy for easier computation
    X_array = np.array(X_with_bias)
    y_array = np.array(y)
    weights_array = np.array(weights)
    
    # Training loop
    for epoch in range(max_epochs):
        converged = True
        
        for i in range(n_samples):
            # Compute prediction: w^T * x
            prediction = np.dot(weights_array, X_array[i])
            
            # Check if prediction is wrong: y * (w^T * x) <= 0
            if y_array[i] * prediction <= 0:
                # Update weights: w = w + learning_rate * y * x
                weights_array += learning_rate * y_array[i] * X_array[i]
                converged = False
        
        # If no updates were made, we've converged
        if converged:
            break
    
    # Generate predictions on training data
    predictions = []
    for i in range(n_samples):
        prediction = np.dot(weights_array, X_array[i])
        predictions.append(1 if prediction > 0 else -1)
    
    return weights_array.tolist(), predictions
