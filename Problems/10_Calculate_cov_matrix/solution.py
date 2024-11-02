def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    n_observations = len(vectors)
    n_features = len(vectors[0])
    covariance_matrix = [[0 for _ in range(n_observations)] for _ in range(n_observations)]

    means = [sum(feature) / n_features for feature in vectors]

    for i in range(n_observations):
        for j in range(i, n_observations):
            covariance = sum((vectors[i][k] - means[i]) * (vectors[j][k] - means[j]) for k in range(n_features)) / (n_features - 1)
            covariance_matrix[i][j] = covariance_matrix[j][i] = covariance

    return covariance_matrix

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
