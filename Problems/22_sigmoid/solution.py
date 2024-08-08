import math

def sigmoid(z: float) -> float:
    result = 1 / (1 + math.exp(-z))
    return round(result, 4)

def test_sigmoid():
    # Test case 1: z = 0
    assert sigmoid(0) == 0.5, "Test case 1 failed"
    
    # Test case 2: z = 1
    assert sigmoid(1) == 0.7311, "Test case 2 failed"
    
    # Test case 3: z = -1
    assert sigmoid(-1) == 0.2689, "Test case 3 failed"

if __name__ == "__main__":
    test_sigmoid()
    print("All sigmoid tests passed.")
