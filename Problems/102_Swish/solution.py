import math

def swish(x: float) -> float:
    """
    Implements the Swish activation function.
    
    Args:
        x: Input value
        
    Returns:
        The Swish activation value
    """
    return x * (1 / (1 + math.exp(-x)))

def test_swish():
    # Test case 1: x = 0
    assert abs(swish(0) - 0) < 1e-6, "Test case 1 failed"
    
    # Test case 2: x = 1
    expected = 1 * (1 / (1 + math.exp(-1)))
    assert abs(swish(1) - expected) < 1e-6, "Test case 2 failed"
    
    # Test case 3: x = -1
    expected = -1 * (1 / (1 + math.exp(1)))
    assert abs(swish(-1) - expected) < 1e-6, "Test case 3 failed"
    
    # Test case 4: large positive number
    x = 10.0
    assert abs(swish(x) - x) < 0.01, "Test case 4 failed"  # Should be close to x
    
    # Test case 5: large negative number
    assert abs(swish(-10.0)) < 0.01, "Test case 5 failed"  # Should be close to 0

if __name__ == "__main__":
    test_swish()
    print("All Swish tests passed.")
    