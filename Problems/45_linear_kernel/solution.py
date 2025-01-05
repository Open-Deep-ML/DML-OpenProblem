import numpy as np

def kernel_function(x1, x2):
    """
    Linear kernel using np.inner for computing pairwise inner products.
    """
    return np.inner(x1, x2)

def test_kernel_function():
    # Test case 1 (1D arrays)
    x1 = np.array([1, 2, 3])
    x2 = np.array([4, 5, 6])
    expected_output = 32  # 1*4 + 2*5 + 3*6
    assert kernel_function(x1, x2) == expected_output, "Test case 1 failed"
    
    # Test case 2 (1D arrays)
    x1 = np.array([0, 1, 2])
    x2 = np.array([3, 4, 5])
    expected_output = 14  # 0*3 + 1*4 + 2*5
    assert kernel_function(x1, x2) == expected_output, "Test case 2 failed"
    
    # Test case 3 (2D arrays - demonstrates difference between np.inner and np.dot)
    x1 = np.array([[1, 2], 
                   [3, 4]])
    x2 = np.array([[5, 6], 
                   [7, 8]])
    # np.inner on 2D arrays will compute pairwise inner products for each row:
    # inner([1,2], [5,6]) = 1*5 + 2*6 = 17
    # inner([1,2], [7,8]) = 1*7 + 2*8 = 23
    # inner([3,4], [5,6]) = 3*5 + 4*6 = 39
    # inner([3,4], [7,8]) = 3*7 + 4*8 = 53
    expected_output = np.array([[17, 23],
                                [39, 53]])
    output = kernel_function(x1, x2)
    assert np.array_equal(output, expected_output), \
           f"Test case 3 failed. Expected:\n{expected_output}\nBut got:\n{output}"
    
    # Test case 4 (2D arrays with different shapes for more variety)
    x1 = np.array([[1, 2, 3],
                   [4, 5, 6]])
    x2 = np.array([[0, 1, 2],
                   [3, 4, 5]])
    # np.inner will compute a (2x2) result:
    # [1,2,3] dot [0,1,2] = 0 + 2 + 6 = 8
    # [1,2,3] dot [3,4,5] = 3 + 8 + 15 = 26
    # [4,5,6] dot [0,1,2] = 0 + 5 + 12 = 17
    # [4,5,6] dot [3,4,5] = 12 + 20 + 30 = 62
    expected_output = np.array([[ 8, 26],
                                [17, 62]])
    output = kernel_function(x1, x2)
    assert np.array_equal(output, expected_output), \
           f"Test case 4 failed. Expected:\n{expected_output}\nBut got:\n{output}"

if __name__ == "__main__":
    test_kernel_function()
    print("All kernel_function tests passed.")
