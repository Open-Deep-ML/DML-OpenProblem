import numpy as np

def linear_kernel(x, y):
    return np.dot(x, y)

def rbf_kernel(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def pegasos_kernel_svm(data, labels, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0):
    n_samples = len(data)
    alphas = np.zeros(n_samples)
    b = 0

    for t in range(1, iterations + 1):
        for i in range(n_samples):
            eta = 1.0 / (lambda_val * t)
            if kernel == 'linear':
                kernel_func = linear_kernel
            elif kernel == 'rbf':
                kernel_func = lambda x, y: rbf_kernel(x, y, sigma)
    
            decision = sum(alphas[j] * labels[j] * kernel_func(data[j], data[i]) for j in range(n_samples)) + b
            if labels[i] * decision < 1:
                alphas[i] += eta * (labels[i] - lambda_val * alphas[i])
                b += eta * labels[i]

    return np.round(alphas, 4).tolist(), np.round(b, 4)


def test_pegasos_kernel_svm():
    # Test case 1: Linear kernel
    data = np.array([[1, 2], [2, 3], [3, 1], [4, 1]])
    labels = np.array([1, 1, -1, -1])
    expected_output = ([100.0, 0.0, -100.0, -100.0], -937.4755)
    assert pegasos_kernel_svm(data, labels, kernel='linear', lambda_val=0.01, iterations=100) == expected_output, "Test case 1 failed"
    
    # Test case 2: RBF kernel
    expected_output = ([100.0, 99.0, -100.0, -100.0], -115.0)
    assert pegasos_kernel_svm(data, labels, kernel='rbf', lambda_val=0.01, iterations=100, sigma=0.5) == expected_output, "Test case 2 failed"

if __name__ == "__main__":
    test_pegasos_kernel_svm()
    print("All pegasos_kernel_svm tests passed.")
