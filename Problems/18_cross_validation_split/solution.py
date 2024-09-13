import numpy as np

def cross_validation_split(data: np.ndarray, k: int, seed=42) -> list:
    np.random.seed(seed)
    np.random.shuffle(data)
    
    n, m = data.shape
    sub_size = int(np.ceil(n / k))
    id_s = np.arange(0, n, sub_size)
    id_e = id_s + sub_size
    if id_e[-1] > n: id_e[-1] = n

    return [[np.concatenate([data[: id_s[i]], data[id_e[i]:]], axis=0).tolist(), data[id_s[i]: id_e[i]].tolist()] for i in range(k)]

def test_cross_validation_split():
    # Test case 1
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    k = 5
    expected_output = [[[[9, 10], [5, 6], [1, 2], [7, 8]], [[3, 4]]], [[[3, 4], [5, 6], [1, 2], [7, 8]], [[9, 10]]], [[[3, 4], [9, 10], [1, 2], [7, 8]], [[5, 6]]], [[[3, 4], [9, 10], [5, 6], [7, 8]], [[1, 2]]], [[[3, 4], [9, 10], [5, 6], [1, 2]], [[7, 8]]]]
    assert cross_validation_split(data, k) == expected_output, "Test case 1 failed"
    
    # Test case 3
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    k = 2
    expected_output = [[[[1, 2], [7, 8]], [[3, 4], [9, 10], [5, 6]]], [[[3, 4], [9, 10], [5, 6]], [[1, 2], [7, 8]]]]
    assert cross_validation_split(data, k) == expected_output, "Test case 2 failed"
    
    # Test case 3
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
    k = 3
    expected_output = [[[[15, 16], [5, 6], [9, 10], [7, 8], [13, 14]], [[3, 4], [11, 12], [1, 2]]], [[[3, 4], [11, 12], [1, 2], [7, 8], [13, 14]], [[15, 16], [5, 6], [9, 10]]], [[[3, 4], [11, 12], [1, 2], [15, 16], [5, 6], [9, 10]], [[7, 8], [13, 14]]]]
    assert cross_validation_split(data, k) == expected_output, "Test case 3 failed"

if __name__ == "__main__":
    test_cross_validation_split()
    print("All cross_validation_split tests passed.")
