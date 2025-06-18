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
