import numpy as np

def im2col(img: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.ndarray:
    """
        Converts a 2D image into a collection of flattened patches.

        Each patch corresponds to a region covered by the convolution kernel determined by the specified stride.

        Padding is assumed to be handled externally.
        Args:
            img (np.ndarray): image to flatten
            kernel (np.ndarray): convolution kernel
            stride (int): step size between patches
        Returns:
            np.ndarray: 2D array where each row is a flattened image patch
    """

    # unpack shapes
    img_h, img_w = img.shape
    kern_h, kern_w = kernel.shape

    # desired output shape ( (input - filter) / stride ) + 1
    out_h = (img_h - kern_h) // stride + 1
    out_w = (img_w - kern_w) // stride + 1

    cols = np.zeros((out_h * out_w, kern_h * kern_w))

    # iterates over every patch (relative to stride), flattens patch into row and inputs into our 2d array as a column
    col_idx = 0
    for y in range(0, out_h * stride, stride):
        for x in range(0, out_w * stride, stride):
            patch = img[y:y + kern_h, x:x + kern_w]
            cols[col_idx, :] = patch.flatten()
            col_idx += 1

    return cols