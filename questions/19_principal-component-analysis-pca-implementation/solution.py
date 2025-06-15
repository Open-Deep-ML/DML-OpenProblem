import numpy as np


def pca(data, k):
    # Standardize the data
    data_standardized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Compute the covariance matrix
    covariance_matrix = np.cov(data_standardized, rowvar=False)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort the eigenvectors by decreasing eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:, idx]

    # Select the top k eigenvectors (principal components)
    principal_components = eigenvectors_sorted[:, :k]

    return np.round(principal_components, 4)
