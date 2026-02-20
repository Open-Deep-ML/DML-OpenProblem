import numpy as np


def simple_conv2d(
    input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int
):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    padded_input = np.pad(
        input_matrix, ((padding, padding), (padding, padding)), mode="constant"
    )
    input_height_padded, input_width_padded = padded_input.shape

    output_height = (input_height_padded - kernel_height) // stride + 1
    output_width = (input_width_padded - kernel_width) // stride + 1

    output_matrix = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            region = padded_input[
                i * stride : i * stride + kernel_height,
                j * stride : j * stride + kernel_width,
            ]
            output_matrix[i, j] = np.sum(region * kernel)

    return output_matrix


def test_simple_conv2d():
    # Test case 1
    input_matrix = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0, 25.0],
        ]
    )
    kernel = np.array(
        [
            [1.0, 2.0],
            [3.0, -1.0],
        ]
    )
    padding, stride = 0, 1
    expected = np.array(
        [
            [16.0, 21.0, 26.0, 31.0],
            [41.0, 46.0, 51.0, 56.0],
            [66.0, 71.0, 76.0, 81.0],
            [91.0, 96.0, 101.0, 106.0],
        ]
    )
    assert np.array_equal(
        expected, simple_conv2d(input_matrix, kernel, padding, stride)
    )

    # Test case 2
    padding, stride = 1, 1
    expected = np.array(
        [
            [-1.0, 1.0, 3.0, 5.0, 7.0, 15.0],
            [-4.0, 16.0, 21.0, 26.0, 31.0, 35.0],
            [1.0, 41.0, 46.0, 51.0, 56.0, 55.0],
            [6.0, 66.0, 71.0, 76.0, 81.0, 75.0],
            [11.0, 91.0, 96.0, 101.0, 106.0, 95.0],
            [42.0, 65.0, 68.0, 71.0, 74.0, 25.0],
        ]
    )
    assert np.array_equal(
        expected, simple_conv2d(input_matrix, kernel, padding, stride)
    )

    # Test case 3
    kernel = np.array(
        [
            [
                1.0,
                2.0,
                3.0,
            ],
            [
                -6.0,
                2.0,
                8.0,
            ],
            [
                5.0,
                2.0,
                3.0,
            ],
        ]
    )
    padding, stride = 0, 1
    expected = np.array(
        [
            [174.0, 194.0, 214.0],
            [274.0, 294.0, 314.0],
            [374.0, 394.0, 414.0],
        ]
    )
    assert np.array_equal(
        expected, simple_conv2d(input_matrix, kernel, padding, stride)
    )

    # Test case 4
    padding, stride = 1, 2
    expected = np.array(
        [
            [51.0, 104.0, 51.0],
            [234.0, 294.0, 110.0],
            [301.0, 216.0, -35.0],
        ]
    )
    assert np.array_equal(
        expected, simple_conv2d(input_matrix, kernel, padding, stride)
    )

    # Test case 5
    input_matrix = np.array(
        [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ]
    )
    kernel = np.array(
        [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ]
    )
    padding, stride = 1, 1
    expected = np.array([[16.0, 28.0, 16.0], [24.0, 42.0, 24.0], [16.0, 28.0, 16.0]])
    assert np.array_equal(
        expected, simple_conv2d(input_matrix, kernel, padding, stride)
    )


if __name__ == "__main__":
    test_simple_conv2d()
    print("All simple_conv2d tests passed.")
