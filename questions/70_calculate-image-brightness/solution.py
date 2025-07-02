def calculate_brightness(img):
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
