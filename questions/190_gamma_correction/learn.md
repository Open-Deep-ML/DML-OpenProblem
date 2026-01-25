## Solution Explanation

In this problem, we are given a grayscale image represented as a 2D matrix and a positive real number `gamma`.  
Each value in the matrix represents a pixel intensity between `0` (black) and `255` (white).

Our goal is to apply **gamma correction** to every pixel in the image and return the corrected image.

---

### Intuition

Human eyes do not perceive brightness linearly.  
Gamma correction is used to adjust image brightness in a nonlinear way so that images look more natural.

- If `gamma > 1`, the image becomes darker.
- If `gamma < 1`, the image becomes brighter.

Each pixel is transformed independently using a mathematical formula.

---

### Mathematical Formula

For each pixel value `p`, the corrected value is computed as:

$$
new\_pixel = 255 \times \left(\frac{p}{255}\right)^\gamma
$$

Where:

- `p` is the original pixel value
- `gamma` is the given correction value
- `new_pixel` is the transformed pixel value

After computing this value:

1. Round it to the nearest integer.
2. Clip it to the range `[0, 255]`.

---

### Step-by-Step Approach

We follow these steps:

#### 1. Validate Input

Before processing, we check:

- The image is not empty.
- All rows have the same length.
- All pixel values are in `[0, 255]`.
- `gamma > 0`.

If any condition fails, return `-1`.

---

#### 2. Normalize Pixel Values

Each pixel `p` is first converted to the range `[0, 1]`:

$$
normalized = \frac{p}{255}
$$

This makes the power operation mathematically stable.

---

#### 3. Apply Gamma Correction

We raise the normalized value to the power `gamma`:

$$
corrected = normalized^\gamma
$$

Then scale it back to `[0, 255]`:

note: maybe not this
$$
scaled = 255 \times corrected
$$

---

#### 4. Round and Clip

The scaled value is:

- Rounded to the nearest integer
- Clipped so that:

$$
0 \le new\_pixel \le 255
$$

This ensures valid pixel values.

---

#### 5. Build Output Image

We repeat the above steps for every pixel and store the results in a new 2D matrix.

This matrix is returned as the final output.

---

### Example Walkthrough

Consider the input:

