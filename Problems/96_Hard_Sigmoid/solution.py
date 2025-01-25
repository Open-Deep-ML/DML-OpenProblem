def hard_sigmoid(x: float) -> float:
    """
    Implements the Hard Sigmoid activation function.
    
    Args:
        x (float): Input value
        
    Returns:
        float: The Hard Sigmoid of the input
    """
    if x <= -2.5:
        return 0.0
    elif x >= 2.5:
        return 1.0
    else:
        return 0.2 * x + 0.5

def test_hard_sigmoid():
    # Test case 1: x <= -2.5
    assert hard_sigmoid(-3.0) == 0.0, "Test case 1 failed"
    
    # Test case 2: x >= 2.5
    assert hard_sigmoid(3.0) == 1.0, "Test case 2 failed"
    
    # Test case 3: -2.5 < x < 2.5
    assert abs(hard_sigmoid(0.0) - 0.5) < 1e-6, "Test case 3 failed"
    assert abs(hard_sigmoid(1.0) - 0.7) < 1e-6, "Test case 4 failed"
    assert abs(hard_sigmoid(-1.0) - 0.3) < 1e-6, "Test case 5 failed"
    
    # Test boundary cases
    assert abs(hard_sigmoid(2.5) - 1.0) < 1e-6, "Test case 6 failed"
    assert abs(hard_sigmoid(-2.5) - 0.0) < 1e-6, "Test case 7 failed"

if __name__ == "__main__":
    test_hard_sigmoid()
    print("All Hard Sigmoid tests passed.")