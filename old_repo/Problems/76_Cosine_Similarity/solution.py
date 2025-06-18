import numpy as np

def cosine_similarity(v1, v2):
    if v1.shape != v2.shape:
        raise ValueError("Arrays must have the same shape")
    
    if v1.size == 0:
        raise ValueError("Arrays cannot be empty")
        
    # Flatten arrays in case of 2D
    v1_flat = v1.flatten()
    v2_flat = v2.flatten()
    
    dot_product = np.dot(v1_flat, v2_flat)
    magnitude1 = np.sqrt(np.sum(v1_flat**2))
    magnitude2 = np.sqrt(np.sum(v2_flat**2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        raise ValueError("Vectors cannot have zero magnitude")
        
    return round(dot_product / (magnitude1 * magnitude2), 3)

def test_cosine_similarity():
    v1_test1 = np.array([1, 2, 3])
    v2_test1 = np.array([2, 4, 6])
    expected1 = 1.0 
    assert cosine_similarity(v1_test1, v2_test1) == expected1, "Test Case Failed"
    
    v1_test2 = np.array([[1, 2], [3, 4]])
    v2_test2 = np.array([[2, 4], [6, 8]])
    expected2 = 1.0
    assert cosine_similarity(v1_test2, v2_test2) == expected2, "Test Case Failed"
    
    v1_test3 = np.array([1, 2, 3])
    v2_test3 = np.array([-1, -2, -3])
    expected3 = -1.0
    assert cosine_similarity(v1_test3, v2_test3) == expected3, "Test Case Failed"
    
    # Test Case : Orthogonal vectors
    v1_test4 = np.array([1, 0])
    v2_test4 = np.array([0, 1])
    expected4 = 0.0
    assert cosine_similarity(v1_test4, v2_test4) == expected4, "Test Case Failed"
    
    # Test Case : Different shapes
    v1_test5 = np.array([1, 2, 3])
    v2_test5 = np.array([1, 2])
    try:
        cosine_similarity(v1_test5, v2_test5)
        assert False, "Test Case Failed"
    except ValueError:
        pass
    
    # Test Case : Zero magnitude vector
    v1_test7 = np.array([0, 0, 0])
    v2_test7 = np.array([1, 2, 3])
    try:
        cosine_similarity(v1_test7, v2_test7)
        assert False, "Test Case Failed"
    except ValueError:
        pass

if __name__ == "__main__":
    test_cosine_similarity()
    print("All Test Cases Passed!")
