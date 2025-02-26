import math

def elu(x: float, alpha: float = 1.0) -> float:
    """
    Compute the ELU activation function.
    
    Args:
        x (float): Input value
        alpha (float): ELU parameter for negative values (default: 1.0)
    
    Returns:
        float: ELU activation value
    """
    return x if x > 0 else alpha * (math.exp(x) - 1)

def test_elu():
    # Test case 1: x = 0
    assert abs(elu(0)) < 1e-10, "Test case 1 failed"
    
    # Test case 2: positive input
    assert abs(elu(1) - 1) < 1e-10, "Test case 2 failed"
    
    # Test case 3: negative input
    assert abs(elu(-1) - (-0.6321205588285577)) < 1e-10, "Test case 3 failed"
    
    # Test case 4: different alpha value
    assert abs(elu(-1, alpha=2.0) - (-1.2642411176571154)) < 1e-10, "Test case 4 failed"
    
    # Test case 5: large positive input
    assert abs(elu(5) - 5) < 1e-10, "Test case 5 failed"
    
    # Test case 6: large negative input
    assert abs(elu(-5) - (-0.9932620530009145)) < 1e-10, "Test case 6 failed"

if __name__ == "__main__":
    test_elu()
    print("All ELU tests passed.")