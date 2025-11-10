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

    # TODO: IM2COL algorithm 
    cols = np.array([])

    return cols