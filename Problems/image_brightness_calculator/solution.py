def calculate_brightness(
    img: list[list[int]],
) -> float:
    # Check if image is empty or has no columns
    if not img or not img[0]:
        return -1
    
    rows, cols = len(img), len(img[0])
    
    # Check if all rows have same length and values are valid
    for row in img:
        if len(row) != cols:
            return -1
        for pixel in row:
            if not 0 <= pixel <= 255:
                return -1
    
    # Calculate average brightness
    total = sum(sum(row) for row in img)
    return round(total / (rows * cols), 2)


def test_calculate_brightness() -> None:
    # Test empty image
    assert calculate_brightness([]) == -1
    
    # Test invalid dimensions
    assert calculate_brightness([[100, 200], [150]]) == -1
    
    # Test invalid pixel values
    assert calculate_brightness([[100, 300]]) == -1
    assert calculate_brightness([[100, -1]]) == -1
    
    # Test valid cases
    assert calculate_brightness([[128]]) == 128.0
    assert calculate_brightness([[100, 200], [50, 150]]) == 125.0
    
if __name__ == "__main__":
    test_calculate_brightness()
    print("All tests passed.")