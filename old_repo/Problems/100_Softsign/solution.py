def softsign(x: float) -> float:
    """
    Implements the Softsign activation function.
    
    Args:
        x (float): Input value
        
    Returns:
        float: The Softsign of the input, calculated as x/(1 + |x|)
    """
    return x / (1 + abs(x))

def test_softsign():
    # Test case 1: x = 0
    assert abs(softsign(0) - 0) < 1e-7, "Test case 1 failed"
    
    # Test case 2: x = 1
    assert abs(softsign(1) - 0.5) < 1e-7, "Test case 2 failed"
    
    # Test case 3: x = -1
    assert abs(softsign(-1) - (-0.5)) < 1e-7, "Test case 3 failed"
    
    # Test case 4: large positive number
    assert abs(softsign(100) - 0.9901) < 1e-4, "Test case 4 failed"
    
    # Test case 5: large negative number
    assert abs(softsign(-100) - (-0.9901)) < 1e-4, "Test case 5 failed"

if __name__ == "__main__":
    test_softsign()
    print("All Softsign tests passed.")
