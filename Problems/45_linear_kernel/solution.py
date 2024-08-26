import numpy as np

def kernel_function(x1, x2):
    return np.inner(x1, x2)

def test_kernel_function():
    # Test case 1
    x1 = np.array([1, 2, 3])
    x2 = np.array([4, 5, 6])
    expected_output = 32
    assert kernel_function(x1, x2) == expected_output, "Test case 1 failed"
    
    # Test case 2
    x1 = np.array([0, 1, 2])
    x2 = np.array([3, 4, 5])
    expected_output = 14
    assert kernel_function(x1, x2) == expected_output, "Test case 2 failed"

if __name__ == "__main__":
    test_kernel_function()
    print("All kernel_function tests passed.")
