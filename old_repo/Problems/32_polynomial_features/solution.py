import numpy as np
from itertools import combinations_with_replacement


def polynomial_features(X, degree):
    n_samples, n_features = np.shape(X)

    # Generate all combinations of feature indices for polynomial terms
    def index_combinations():
        combs = [
            combinations_with_replacement(range(n_features), i)
            for i in range(0, degree + 1)
        ]
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
    expected_output = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 6.0, 9.0],
            [1.0, 3.0, 4.0, 9.0, 12.0, 16.0],
            [1.0, 5.0, 6.0, 25.0, 30.0, 36.0],
        ]
    )
    assert np.allclose(polynomial_features(X, degree), expected_output), (
        "Test case 1 failed"
    )

    # Test case 2
    X = np.array([[1, 2], [3, 4], [5, 6]])
    degree = 3
    expected_output = np.array(
        [
            [1.0, 1.0, 2.0, 1.0, 2.0, 4.0, 1.0, 2.0, 4.0, 8.0],
            [1.0, 3.0, 4.0, 9.0, 12.0, 16.0, 27.0, 36.0, 48.0, 64.0],
            [1.0, 5.0, 6.0, 25.0, 30.0, 36.0, 125.0, 150.0, 180.0, 216.0],
        ]
    )
    assert np.allclose(polynomial_features(X, degree), expected_output), (
        "Test case 2 failed"
    )

    # Test case 3
    X = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 9]])
    degree = 3
    expected_output = np.array(
        [
            [
                1.0,
                1.0,
                2.0,
                3.0,
                1.0,
                2.0,
                3.0,
                4.0,
                6.0,
                9.0,
                1.0,
                2.0,
                3.0,
                4.0,
                6.0,
                9.0,
                8.0,
                12.0,
                18.0,
                27.0,
            ],
            [
                1.0,
                3.0,
                4.0,
                5.0,
                9.0,
                12.0,
                15.0,
                16.0,
                20.0,
                25.0,
                27.0,
                36.0,
                45.0,
                48.0,
                60.0,
                75.0,
                64.0,
                80.0,
                100.0,
                125.0,
            ],
            [
                1.0,
                5.0,
                6.0,
                9.0,
                25.0,
                30.0,
                45.0,
                36.0,
                54.0,
                81.0,
                125.0,
                150.0,
                225.0,
                180.0,
                270.0,
                405.0,
                216.0,
                324.0,
                486.0,
                729.0,
            ],
        ]
    )
    assert np.allclose(polynomial_features(X, degree), expected_output), (
        "Test case 3 failed"
    )


if __name__ == "__main__":
    test_polynomial_features()
    print("All polynomial_features tests passed.")
