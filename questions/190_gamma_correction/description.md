## Problem

You are given a grayscale image represented as a 2D matrix and a gamma value.

Each pixel ranges from 0 to 255.

Apply gamma correction using:

new_pixel = 255 * (pixel / 255) ^ gamma

Round to nearest integer.

Return the corrected image.

### Edge Cases

Return -1 if:

- Image is empty
- Rows are inconsistent
- Pixel values are invalid
- Gamma <= 0

