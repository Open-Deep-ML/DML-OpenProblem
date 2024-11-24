import numpy as np

def calculate_dot_product(vec1, vec2):
    """
    Calculate the dot product of two vectors.

    Args:
        vec1 (numpy.ndarray): 1D array representing the first vector.
        vec2 (numpy.ndarray): 1D array representing the second vector.

    Returns:
        float: Dot product of the two vectors.
    """
    return np.dot(vec1, vec2)

def test_calculate_dot_product():
    # Test case 1: Positive numbers
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([4, 5, 6])
    expected_output1 = 32
    assert calculate_dot_product(vec1, vec2) == expected_output1, "Test case 1 failed"
    
    # Test case 2: Include negative numbers
    vec1 = np.array([-1, 2, 3])
    vec2 = np.array([4, -5, 6])
    expected_output2 = 4
    assert calculate_dot_product(vec1, vec2) == expected_output2, "Test case 2 failed"
    
    # Test case 3: Orthogonal vectors
    vec1 = np.array([1, 0])
    vec2 = np.array([0, 1])
    expected_output3 = 0
    assert calculate_dot_product(vec1, vec2) == expected_output3, "Test case 3 failed"
    
    # Test case 4: All zeros
    vec1 = np.array([0, 0, 0])
    vec2 = np.array([0, 0, 0])
    expected_output4 = 0
    assert calculate_dot_product(vec1, vec2) == expected_output4, "Test case 4 failed"
    
    # Test case 5: Scalars (single-element vectors)
    vec1 = np.array([7])
    vec2 = np.array([3])
    expected_output5 = 21
    assert calculate_dot_product(vec1, vec2) == expected_output5, "Test case 5 failed"

if __name__ == "__main__":
    test_calculate_dot_product()
    print("All dot product test cases passed.")
