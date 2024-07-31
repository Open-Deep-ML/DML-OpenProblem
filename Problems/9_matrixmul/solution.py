def matrixmul(a: list[list[int | float]], b: list[list[int | float]]) -> list[list[int | float]]:
    if len(a[0]) != len(b):
        return -1
    
    vals = []
    for i in range(len(a)):
        hold = []
        for j in range(len(b[0])):
            val = 0
            for k in range(len(b)):
                val += a[i][k] * b[k][j]
                           
            hold.append(val)
        vals.append(hold)

    return vals

def test_matrixmul() -> None:
    # Test cases for matrixmul function

    # Test case 1
    a = [[1, 2, 3], [2, 3, 4], [5, 6, 7]]
    b = [[3, 2, 1], [4, 3, 2], [5, 4, 3]]
    assert matrixmul(a, b) == [[26, 20, 14], [38, 29, 20], [74, 56, 38]]

    # Test case 2
    a = [[0, 0], [2, 4], [1, 2]]
    b = [[0, 0], [2, 4]]
    assert matrixmul(a, b) == [[0, 0], [8, 16], [4, 8]]

    # Test case 3
    a = [[0, 0], [2, 4], [1, 2]]
    b = [[0, 0, 1], [2, 4, 1], [1, 2, 3]]
    assert matrixmul(a, b) == -1

if __name__ == "__main__":
    test_matrixmul()
    print("All matrixmul tests passed.")
