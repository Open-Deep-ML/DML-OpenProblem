import numpy as np

def partial_pivoting(A_aug: np.array, row_num: int, col_num: int) -> np.array:
    
    rows, cols = A_aug.shape
    
    max_row = row_num
    max_val = abs(A_aug[row_num,col_num])
    
    for i in range(row_num, rows):
        current_val = abs(A_aug[i, col_num])
        if current_val > max_val:
            max_val = current_val
            max_row = i
    
    if max_row != row_num:
        A_aug[[row_num, max_row]] = A_aug[[max_row, row_num]]
            
    return A_aug

def gaussian_elimination(A: np.array, b: np.array) -> np.array:
    
    # get original matrix dimensions to avoid overwriting source later in augmented matrix
    rows, cols = A.shape
    
    # create augmented matrix
    A_aug = np.hstack((A,b.reshape(-1,1)))
    
    # Elimination
    # loop over rows
    for i in range(rows-1):
        A_aug = partial_pivoting(A_aug, i, i)
        # apply elimination to all rows below the current row
        for j in range(i+1, rows):
            A_aug[j, i:] -= (A_aug[j,i] / A_aug[i,i]) * A_aug[i, i:]

    x = np.zeros_like(b, dtype=float)
    
    # Backsubtitution
    for i in range(rows-1,-1,-1):
        x[i] = (A_aug[i,-1] - np.dot(A_aug[i,i+1:cols],x[i+1:]))/A_aug[i,i]
    
    return x

def test_gaussian_elimination():
    # Test case 1: basic test
    A_1 = np.array([[2,8,4],[2,5,1],[4,10,-1]], dtype=float)
    b_1 = np.array([2,5,1], dtype=float)
    expected_1 = np.array([11., -4.,  3.])
    output_1 = gaussian_elimination(A_1, b_1)
    assert np.allclose(output_1, expected_1, atol=0.01), f"Test case 1 failed: expected {expected_1}, got {output_1}"

    # Test case 2: testing a zero pivot
    A_2 = np.array([
        [0, 2, 1, 0, 0, 0, 0],
        [2, 6, 2, 1, 0, 0, 0],
        [1, 2, 7, 2, 1, 0, 0],
        [0, 1, 2, 8, 2, 1, 0],
        [0, 0, 1, 2, 9, 2, 1],
        [0, 0, 0, 1, 2, 10, 2],
        [0, 0, 0, 0, 1, 2, 11]
    ], dtype=float)
    b_2 = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)
    expected_2 = np.array([-0.4894027, 0.36169985, 0.2766003, 0.25540569, 0.31898951, 0.40387497, 0.53393278])
    output_2 = gaussian_elimination(A_2, b_2)
    assert np.allclose(output_2, expected_2, atol=0.01), f"Test case 2 failed: expected {expected_2}, got {output_2}"

    # Test case 3: Multiple feature inputs
    A_3 = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
    b_3 = np.array([8, -11, -3], dtype=float)
    expected_3 = np.array([2., 3., -1.])
    output_3 = gaussian_elimination(A_3, b_3)
    assert np.allclose(output_3, expected_3, atol=0.01), f"Test case 3 failed: expected {expected_3}, got {output_3}"

if __name__ == "__main__":
    test_gaussian_elimination()
    print("All Gaussian Elimination tests passed.")
