import math

def softplus(x: float) -> float:
    """
    Compute the softplus activation function.
    
    Args:
        x: Input value
        
    Returns:
        The softplus value: log(1 + e^x)
    """
    # To prevent overflow for large positive values
    if x > 100:
        return x
    # To prevent underflow for large negative values
    if x < -100:
        return 0.0
    
    return math.log(1.0 + math.exp(x))

def test_softplus():
    # Test case 1: x = 0
    assert abs(softplus(0) - math.log(2)) < 1e-6, "Test case 1 failed"
    
    # Test case 2: large positive number
    assert abs(softplus(100) - 100) < 1e-6, "Test case 2 failed"
    
    # Test case 3: large negative number
    assert abs(softplus(-100)) < 1e-6, "Test case 3 failed"
    
    # Test case 4: positive number
    assert abs(softplus(2) - 2.1269280110429727) < 1e-6, "Test case 4 failed"
    
    # Test case 5: negative number
    assert abs(softplus(-2) - 0.12692801104297272) < 1e-6, "Test case 5 failed"

if __name__ == "__main__":
    test_softplus()
    print("All Softplus tests passed.")