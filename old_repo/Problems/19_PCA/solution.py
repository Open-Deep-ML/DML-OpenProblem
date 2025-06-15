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
    eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:, idx]
    
    # Select the top k eigenvectors (principal components)
    principal_components = eigenvectors_sorted[:, :k]
    
    return np.round(principal_components, 4).tolist()

def test_pca():
    # Test case 1
    data = np.array([[4, 2, 1], [5, 6, 7], [9, 12, 1], [4, 6, 7]])
    k = 2
    expected_output = [[0.6855, 0.0776], [0.6202, 0.4586], [-0.3814, 0.8853]]
    assert pca(data, k) == expected_output, "Test case 1 failed"

    # Test case 2
    data = np.array([[1, 2], [3, 4], [5, 6]])
    k = 1
    expected_output = [[0.7071], [0.7071]]
    assert pca(data, k) == expected_output, "Test case 2 failed"

if __name__ == "__main__":
    test_pca()
    print("All pca tests passed.")
