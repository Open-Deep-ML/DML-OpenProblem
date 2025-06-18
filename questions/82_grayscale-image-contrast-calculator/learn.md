## Calculating Contrast of a Grayscale Image

Contrast in a grayscale image refers to the difference in luminance or color that makes an object distinguishable. Here are methods to calculate contrast:

### 1. Basic Contrast Calculation

The simplest way to define the contrast of a grayscale image is by using the difference between the maximum and minimum pixel values:

$$
\text{Contrast} = \max(I) - \min(I)
$$

### 2. RMS Contrast

Root Mean Square (RMS) contrast considers the standard deviation of pixel intensities:

$$
\text{RMS Contrast} = \frac{\sigma}{\mu}
$$

### 3. Michelson Contrast

Michelson contrast is defined as:

$$
C = \frac{I_{\text{max}} - I_{\text{min}}}{I_{\text{max}} + I_{\text{min}}}
$$

### Example Calculation

For a grayscale image with pixel values ranging from 50 to 200:

1. **Maximum Pixel Value**: 200  
2. **Minimum Pixel Value**: 50  
3. **Contrast Calculation**:

$$
\text{Contrast} = 200 - 50 = 150
$$

### Applications

Calculating contrast is crucial in:

- Image quality assessment
- Preprocessing in computer vision
- Enhancing visibility in images
- Object detection and analysis
