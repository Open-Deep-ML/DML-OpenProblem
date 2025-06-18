import numpy as np

def rref(matrix):
    # Convert to float for division operations
    A = matrix.astype(np.float32)
    n, m = A.shape
    
    for i in range(n):
        if A[i, i] == 0:
            nonzero_rel_id = np.nonzero(A[i:, i])[0]
            if len(nonzero_rel_id) == 0: continue
            
            A[i] = A[i] + A[nonzero_rel_id[0] + i]

        A[i] = A[i] / A[i, i]
        for j in range(n):
            if i != j:
                A[j] -= A[j, i] * A[i]

    return A

def test_rref():
    # Test case 1
    matrix = np.array([
        [1, 2, -1, -4],
        [2, 3, -1, -11],
        [-2, 0, -3, 22]
    ])
    expected_output = np.array([
        [ 1.,  0.,  0., -8.],
        [ 0.,  1.,  0.,  1.],
        [-0., -0.,  1., -2.]
    ])
    assert np.allclose(rref(matrix), expected_output), "Test case 1 failed"
    
    # Test case 2
    matrix = np.array([
        [2, 4, -2],
        [4, 9, -3],
        [-2, -3, 7]
    ])
    expected_output = np.array([
        [ 1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [ 0.,  0.,  1.]
    ])
    assert np.allclose(rref(matrix), expected_output), "Test case 2 failed"
    
    # Test case 3
    matrix = np.array([
        [0, 2, -1, -4],
        [2, 0, -1, -11],
        [-2, 0, 0, 22]
    ])
    expected_output = np.array([
        [ 1.,  0.,  0., -11.],
        [-0.,  1.,  0., -7.5],
        [-0., -0.,  1., -11.]
    ])
    assert np.allclose(rref(matrix), expected_output), "Test case 3 failed"
    
   # Test case 4
    matrix = np.array([
        [1, 2, -1],
        [2, 4, -1],
        [-2, -4, -3]])
    expected_output = np.array([
        [ 1.,  2.,  0.],
        [ 0.,  0.,  0.],
        [-0., -0.,  1.]
    ])
    assert np.allclose(rref(matrix), expected_output), "Test case 4 failed"

if __name__ == "__main__":
    test_rref()
    print("All rref tests passed.")
