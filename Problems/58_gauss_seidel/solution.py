import numpy as np

def gauss_seidel_it(A:np.array, b:np.array, x: np.array) -> np.array:
    
    rows, cols = A.shape
    
    for i in range(rows):
        x_new = b[i]
        for j in range(cols):
            if i != j:
                x_new -= A[i,j] * x[j]
        x[i] = x_new/A[i,i]
    
    return x

def gauss_seidel(A:np.array, b:np.array, n: int, x_ini: np.array=None) -> np.array:
    
    x = x_ini or np.zeros_like(b)
    
    for _ in range(n):
        x = gauss_seidel_it(A,b,x)
        
    return x

def test_gauss_seidel():
    
    # Test case 1: basic test
    A_1 = np.array([[4, 1, 2],[3, 5, 1],[1, 1, 3]], dtype=float)
    b_1 = np.array([4,7,3], dtype=float)
    n = 5
    expected_1 = np.array([0.5008, 0.99968, 0.49984])
    output_1 = gauss_seidel(A_1, b_1, n)
    assert np.allclose(output_1, expected_1, atol=0.01), f"Test case 1 failed: expected {expected_1}, got {output_1}"

    # Test case 2: testing a zero pivot
    A_2 = np.array([[4, -1, 0, 1],
                [-1, 4, -1, 0],
                [0, -1, 4, -1],
                [1, 0, -1, 4]], dtype=float)
    b_2 = np.array([15, 10, 10, 15], dtype=float)
    n = 1
    expected_2 = np.array([3.75, 3.4375, 3.359375, 3.65234375])
    output_2 = gauss_seidel(A_2, b_2, n)
    assert np.allclose(output_2, expected_2, atol=0.01), f"Test case 2 failed: expected {expected_2}, got {output_2}"

    # Test case 3: Multiple feature inputs
    A_3 = np.array([[10, -1, 2],
                [-1, 11, -1],
                [2, -1, 10]], dtype=float)
    b_3 = np.array([6, 25, -11], dtype=float)
    n = 100
    expected_3 = np.array([1.04326923, 2.26923077, -1.08173077])
    output_3 = gauss_seidel(A_3, b_3, n)
    assert np.allclose(output_3, expected_3, atol=0.01), f"Test case 3 failed: expected {expected_3}, got {output_3}"

if __name__ == "__main__":
    test_gauss_seidel()
    print("All Gauss-Seidel tests passed.")
