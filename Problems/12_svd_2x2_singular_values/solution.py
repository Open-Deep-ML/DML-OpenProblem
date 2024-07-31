import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    A_T_A = A.T @ A
    theta = 0.5 * np.arctan2(2 * A_T_A[0, 1], A_T_A[0, 0] - A_T_A[1, 1])
    j = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    A_prime = j.T @ A_T_A @ j 
    
    # Calculate singular values from the diagonalized A^TA (approximation for 2x2 case)
    singular_values = np.sqrt(np.diag(A_prime))

    # Process for AA^T, if needed, similar to A^TA can be added here for completeness
    
    return j, singular_values, j.T

def test_svd_2x2_singular_values() -> None:
    # Test cases for svd_2x2_singular_values function

    # Test case 1
    A = np.array([[2, 1], [1, 2]])
    expected_output = (np.array([[ 0.70710678, -0.70710678],
                                 [ 0.70710678,  0.70710678]]), 
                       np.array([3., 1.]), 
                       np.array([[ 0.70710678,  0.70710678],
                                 [-0.70710678,  0.70710678]]))
    output = svd_2x2_singular_values(A)
    assert np.allclose(output[0], expected_output[0]) and \
           np.allclose(output[1], expected_output[1]) and \
           np.allclose(output[2], expected_output[2]), "Test case 1 failed"

    # Test case 2
    A = np.array([[1, 2], [3, 4]])
    expected_output = (np.array([[ 0.57604844, -0.81741556],
                                 [ 0.81741556,  0.57604844]]), 
                       np.array([5.4649857 , 0.36596619]), 
                       np.array([[ 0.57604844,  0.81741556],
                                 [-0.81741556,  0.57604844]]))
    output = svd_2x2_singular_values(A)
    assert np.allclose(output[0], expected_output[0]) and \
           np.allclose(output[1], expected_output[1]) and \
           np.allclose(output[2], expected_output[2]), "Test case 2 failed"

if __name__ == "__main__":
    test_svd_2x2_singular_values()
    print("All svd_2x2_singular_values tests passed.")
