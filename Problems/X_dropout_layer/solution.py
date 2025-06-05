import numpy as np
import pytest

class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer."""
        if p < 0 or p >= 1:
            raise ValueError("Dropout rate must be between 0 and 1 (exclusive)")
        
        self.p = p
        self.mask = None
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of the dropout layer."""
        if not training:
            return x
                
        self.mask = np.random.binomial(1, 1 - self.p, x.shape)
        
        return x * self.mask / (1 - self.p)
        
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass of the dropout layer."""
        if self.mask is None:
            raise ValueError("Forward pass must be called before backward pass")
            
        return grad * self.mask

def test_dropout_layer():
    np.random.seed(42)
    
    # Test case 1: 2D input
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    grad = np.array([[0.5, 0.2], [1.0, 2.0]])
    p = 0.2


    dropout = DropoutLayer(p)

    output_train = dropout.forward(x, training=True)
    expected_output = np.array([[1.25, 0.0], [3.75, 5.0]])
    assert np.array_equal(output_train, expected_output), "Dropout not applied correctly"

    output_inference = dropout.forward(x, training=False)
    assert np.array_equal(output_inference, x), "Inference mode not working correctly"

    grad_back = dropout.backward(grad)
    expected_grad_back = np.array([[0.5, 0.0], [1.0, 2.0]])
    assert grad_back.shape == x.shape, "Gradient shape mismatch in test case 2"
    assert (grad_back == expected_grad_back).all(), "Gradient not applied correctly"
    
    
    # Test case 2: Mask is different
    x = np.ones((1000, 1000))

    dropout = DropoutLayer(p)

    _ = dropout.forward(x, training=True)
    mask1 = dropout.mask.copy()
    _ = dropout.forward(x, training=True)
    mask2 = dropout.mask.copy()
    assert not np.array_equal(mask1, mask2), "Mask is the same"


    # Test case 2: Expected value preservation
    x = np.ones((1000, 1000))
    p = 0.3
    dropout = DropoutLayer(p)
    output_train = dropout.forward(x, training=True)
    mean_output = np.mean(output_train)
    assert abs(mean_output - 1.0) < 0.1, "Expected value not preserved"


    # Test case 3: Invalid dropout rate
    p = 1.5
    with pytest.raises(ValueError):
        dropout = DropoutLayer(p)
    p = -0.5
    with pytest.raises(ValueError):
        dropout = DropoutLayer(p)

if __name__ == "__main__":
    test_dropout_layer()
    print("All dropout layer tests passed.")
