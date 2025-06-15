def relu(z: float) -> float:
    return max(0, z)

def test_relu():
    # Test case 1: z = 0
    assert relu(0) == 0, "Test case 1 failed"
    
    # Test case 2: z = 1
    assert relu(1) == 1, "Test case 2 failed"
    
    # Test case 3: z = -1
    assert relu(-1) == 0, "Test case 3 failed"

if __name__ == "__main__":
    test_relu()
    print("All ReLU tests passed.")