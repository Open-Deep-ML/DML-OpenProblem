import numpy as np


def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	return [list(x) for x in np.cov(np.array(vectors))]


def test_calculate_covariance_matrix() -> None:
    # Test cases for calculate_covariance_matrix function

    # Test case 1
    vectors = [[1, 2, 3], [4, 5, 6]]
    assert calculate_covariance_matrix(vectors) == [[1.0, 1.0], [1.0, 1.0]]

    # Test case 2
    vectors = [[1, 5, 6], [2, 3, 4], [7, 8, 9]]
    assert calculate_covariance_matrix(vectors) == [[7.0, 2.5, 2.5], [2.5, 1.0, 1.0], [2.5, 1.0, 1.0]]


if __name__ == "__main__":
    test_calculate_covariance_matrix()
    print("All calculate_covariance_matrix tests passed.")
