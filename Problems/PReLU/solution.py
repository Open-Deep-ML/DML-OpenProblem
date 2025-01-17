def prelu(x: float, alpha: float = 0.25) -> float:
    """
    Implements the PReLU (Parametric ReLU) activation function.
    
    Args:
        x: Input value
        alpha: Slope parameter for negative values (default: 0.25)
    
    Returns:
        float: PReLU activation value
    """
    return x if x > 0 else alpha * x

def test_prelu() -> None:
    # Test positive input (should behave like regular ReLU)
    assert prelu(2.0) == 2.0, "Test failed for positive input"
    
    # Test zero input
    assert prelu(0.0) == 0.0, "Test failed for zero input"
    
    # Test negative input with default alpha
    assert prelu(-2.0) == -0.5, "Test failed for negative input with default alpha"
    
    # Test negative input with custom alpha
    assert prelu(-2.0, alpha=0.1) == -0.2, "Test failed for negative input with custom alpha"
    
    # Test with alpha = 0 (behaves like ReLU)
    assert prelu(-2.0, alpha=0.0) == 0.0, "Test failed for ReLU behavior"
    
    # Test with alpha = 1 (behaves like linear function)
    assert prelu(-2.0, alpha=1.0) == -2.0, "Test failed for linear behavior"

if __name__ == "__main__":
    test_prelu()
    print("All PReLU tests passed.")
