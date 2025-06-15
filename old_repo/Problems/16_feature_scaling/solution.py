import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    # Standardization
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    
    # Min-Max Normalization
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    normalized_data = (data - min_val) / (max_val - min_val)
    
    # Rounding to four decimal places and converting to lists
    return np.round(standardized_data, 4).tolist(), np.round(normalized_data, 4).tolist()

def test_feature_scaling() -> None:
    # Test case 1
    data = np.array([[1, 2], [3, 4], [5, 6]])
    standardized, normalized = feature_scaling(data)
    assert standardized == [[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]], "Test case 1 failed"
    assert normalized == [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], "Test case 1 failed"

if __name__ == "__main__":
    test_feature_scaling()
    print("All feature_scaling tests passed.")
