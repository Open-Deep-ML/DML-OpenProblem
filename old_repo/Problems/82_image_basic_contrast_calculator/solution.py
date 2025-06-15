import numpy as np

def calculate_contrast(img):
    """
    Calculate the contrast of a grayscale image.

    Args:
        img (numpy.ndarray): 2D array representing a grayscale image with pixel values between 0 and 255.

    Returns:
        float: Contrast value rounded to 3 decimal places.
    """
    
    # Find the maximum and minimum pixel values
    max_pixel = np.max(img)
    min_pixel = np.min(img)
    
    # Calculate contrast
    contrast = max_pixel - min_pixel
    
    return round(float(contrast), 3)

def test_calculate_contrast():
    # Test case 1: Simple gradient
    img1 = np.array([[0, 50], [200, 255]])
    expected_output1 = 255
    assert calculate_contrast(img1) == expected_output1, "Test case 1 failed"
    
    # Test case 2: Uniform image (all pixels same)
    img2 = np.array([[128, 128], [128, 128]])
    expected_output2 = 0
    assert calculate_contrast(img2) == expected_output2, "Test case 2 failed"
    
    # Test case 3: All black
    img3 = np.zeros((10, 10), dtype=np.uint8)
    expected_output3 = 0
    assert calculate_contrast(img3) == expected_output3, "Test case 3 failed"
    
    # Test case 4: All white
    img4 = np.ones((10, 10), dtype=np.uint8) * 255
    expected_output4 = 0
    assert calculate_contrast(img4) == expected_output4, "Test case 4 failed"
    
    # Test case 5: Random values
    img5 = np.array([[10, 20, 30], [40, 50, 60]])
    expected_output5 = 50
    assert calculate_contrast(img5) == expected_output5, "Test case 5 failed"

if __name__ == "__main__":
    test_calculate_contrast()
    print("All contrast test cases passed.")