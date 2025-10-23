import numpy as np

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
