import numpy as np

def translate_object(points: list[list[float]], tx: float, ty: float) -> list[list[float]]:
    """
    Translate a set of 2D points by tx and ty.
    
    :param points: List of [x, y] coordinates representing the object
    :param tx: Translation in x-direction
    :param ty: Translation in y-direction
    :return: List of translated [x, y] coordinates
    """
    
    # Translation matrix
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    
    # Convert points to homogeneous coordinates
    homogeneous_points = np.hstack([np.array(points), np.ones((len(points), 1))])
    
    # Translation
    translated_points = np.dot(homogeneous_points, translation_matrix.T)
    
    # Convert back to 2D coordinates and to list
    return translated_points[:, :2].tolist()

def test_translate_object() -> None:
    # Test cases for translate_object function
    
    # Test case 1 (triangle)
    triangle = [[0, 0], [1, 0], [0.5, 1]]
    tx, ty = 2, 3
    translated_triangle = translate_object(triangle, tx, ty)
    expected_result = [[2.0, 3.0], [3.0, 3.0], [2.5, 4.0]]
    assert translated_triangle == expected_result, f"Expected {expected_result}, but got {translated_triangle}"
    
    # Test case 2 (square)
    square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    tx, ty = -1, 2
    translated_square = translate_object(square, tx, ty)
    expected_result = [[-1.0, 2.0], [0.0, 2.0], [0.0, 3.0], [-1.0, 3.0]]
    assert translated_square == expected_result, f"Expected {expected_result}, but got {translated_square}"

if __name__ == "__main__":
    test_translate_object()
    print("All tests passed.")