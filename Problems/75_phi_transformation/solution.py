import numpy as np

def phi_transform(data: list[float], degree: int,) -> list[list[float]]:
    if degree < 0 or not data:
        return []
    return np.array([[x ** i for i in range(degree + 1)] for x in data]).tolist()

def test_phi_transform() -> None:
    assert phi_transform([], 2) == [], 'Test Case 1 Failed.'

    assert phi_transform([1.0, 2.0], -1) == [], 'Test Case 2 Failed.'

    data: list[float] = [1.0, 2.0]
    degree: int = 2
    assert phi_transform(data, degree) == [
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 4.0],
    ], 'Test Case 3 Failed.'

    data: list[float] = [1.0, 3.0]
    degree: int = 3
    assert phi_transform(data, degree) == [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 3.0, 9.0, 27.0],
    ], 'Test Case 4 Failed.'

    data: list[float] = [2.0]
    degree: int = 4
    assert phi_transform(data, degree) == [
        [1.0, 2.0, 4.0, 8.0, 16.0]
    ], 'Test Case 5 Failed.'

    

if __name__ == "__main__":
    test_phi_transform()
    print("All tests passed.")
