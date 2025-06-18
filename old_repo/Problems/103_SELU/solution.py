import math

def selu(x: float) -> float:
    """
    Implements the SELU (Scaled Exponential Linear Unit) activation function.
    
    Args:
        x: Input value
        
    Returns:
        SELU activation value
    """
    # Standard SELU parameters
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    
    if x > 0:
        return scale * x
    return scale * alpha * (math.exp(x) - 1)

def test_selu():
    # Test positive input
    assert abs(selu(1.0) - 1.0507009873554804) < 1e-7, "Test case 1 failed"
    
    # Test zero input
    assert abs(selu(0.0) - 0.0) < 1e-7, "Test case 2 failed"
    
    # Test negative input
    assert abs(selu(-1.0) - (-1.1113307)) < 1e-6, "Test case 3 failed"
    
    # Test large positive input
    assert abs(selu(5.0) - 5.2535049) < 1e-6, "Test case 4 failed"
    
    # Test large negative input
    assert abs(selu(-5.0) - (-1.7462534)) < 1e-6, "Test case 5 failed"

if __name__ == "__main__":
    test_selu()
    print("All SELU tests passed.")