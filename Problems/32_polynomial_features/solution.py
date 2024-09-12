import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    n_samples, n_features = np.shape(X)

    # Generate all combinations of feature indices for polynomial terms
    def index_combinations():
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs
    
    combinations = index_combinations()
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))
    
    # Compute polynomial features
    for i, index_combs in enumerate(combinations):  
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new

def test_polynomial_features():
    # Test case 1
    X = np.array([[2, 3], [3, 4], [5, 6]])
    degree = 2
    expected_output = np.array([
        [ 1.,  2.,  3.,  4.,  6.,  9.],
        [ 1.,  3.,  4.,  9., 12., 16.],
        [ 1.,  5.,  6., 25., 30., 36.]
    ])
    assert np.allclose(polynomial_features(X, degree), expected_output), "Test case 1 failed"
    
    # Test case 2
    X = np.array([[1, 2], [3, 4], [5, 6]])
    degree = 3
    expected_output = np.array([
        [  1.,   1.,   2.,   1.,   2.,   4.,   1.,   2.,   4.,   8.],
        [  1.,   3.,   4.,   9.,  12.,  16.,  27.,  36.,  48.,  64.],
        [  1.,   5.,   6.,  25.,  30.,  36., 125., 150., 180., 216.]
    ])
    assert np.allclose(polynomial_features(X, degree), expected_output), "Test case 2 failed"
    
    # Test case 3
    X = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 9]])
    degree = 3
    expected_output = np.array([
        [  1.,   1.,   2.,   3.,   1.,   2.,   3.,   4.,   6.,   9.,   1.,   2.,   3.,   4.,   6.,   9.,   8.,  12.,  18.,  27.],
        [  1.,   3.,   4.,   5.,   9.,  12.,  15.,  16.,  20.,  25.,  27.,  36.,  45.,  48.,  60.,  75.,  64.,  80., 100., 125.],
        [  1.,   5.,   6.,   9.,  25.,  30.,  45.,  36.,  54.,  81., 125., 150., 225., 180., 270., 405., 216., 324., 486., 729.]
    ])
    assert np.allclose(polynomial_features(X, degree), expected_output), "Test case 3 failed"

if __name__ == "__main__":
    test_polynomial_features()
    print("All polynomial_features tests passed.")
