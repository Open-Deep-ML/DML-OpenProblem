import numpy as np

def cross_validation_split(data, k, seed=42):
    np.random.seed(seed)
    np.random.shuffle(data)  # Shuffle the data if desired
    fold_size = len(data) // k
    folds = []
    
    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size if i != k-1 else len(data)
        test = data[start:end]
        train = np.concatenate([data[:start], data[end:]])
        folds.append([train.tolist(), test.tolist()])
    
    return folds

def test_cross_validation_split():
    # Test case 1
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    k = 5
    expected_output = [[[[9, 10], [5, 6], [1, 2], [7, 8]], [[3, 4]]], [[[3, 4], [5, 6], [1, 2], [7, 8]], [[9, 10]]], [[[3, 4], [9, 10], [1, 2], [7, 8]], [[5, 6]]], [[[3, 4], [9, 10], [5, 6], [7, 8]], [[1, 2]]], [[[3, 4], [9, 10], [5, 6], [1, 2]], [[7, 8]]]]
    assert cross_validation_split(data, k) == expected_output, "Test case 1 failed"

if __name__ == "__main__":
    test_cross_validation_split()
    print("All cross_validation_split tests passed.")
