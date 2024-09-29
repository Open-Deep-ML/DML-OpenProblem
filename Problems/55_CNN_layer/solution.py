import numpy as np

def cnn_forward(input_matrix: np.ndarray, kernel: np.ndarray, bias: float = 0.0, stride: int = 1, padding: int = 0) -> np.ndarray:
    # Add padding to the input matrix
    if padding > 0:
        input_matrix = np.pad(input_matrix, pad_width=padding, mode='constant', constant_values=0)
    
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate output dimensions
    output_height = ((input_height - kernel_height) // stride) + 1
    output_width = ((input_width - kernel_width) // stride) + 1

    output = np.zeros((output_height, output_width))
    for i in range(output_height):
        for j in range(output_width):
            # Calculate the starting indices of the current patch
            start_i = i * stride
            start_j = j * stride
            # Extract the current input sub-matrix
            current_patch = input_matrix[start_i:start_i + kernel_height, start_j:start_j + kernel_width]
            # Element-wise multiplication and sum, add bias
            output[i, j] = np.sum(current_patch * kernel) + bias
    return output

def test_cnn_forward():
    # Test case 1: Original example with padding and stride
    input_matrix_1 = np.array([
        [1, 2, 0],
        [0, 1, 2],
        [2, 1, 0]
    ])
    kernel_1 = np.array([
        [1, 0],
        [0, -1]
    ])
    bias_1 = 0.0
    stride_1 = 1
    padding_1 = 1

    # Corrected expected output after recalculations
    expected_output_1 = np.array([
        [-1., -2.,  0.,  0.],
        [ 0.,  0.,  0.,  0.],
        [-2., -1.,  1.,  2.],
        [ 0.,  2.,  1.,  0.]
    ])
    output_1 = cnn_forward(input_matrix_1, kernel_1, bias_1, stride_1, padding_1)
    assert np.array_equal(output_1, expected_output_1), f"Test case 1 failed: Expected output {expected_output_1}, got {output_1}"

    # Test case 2: Larger input with stride 2
    input_matrix_2 = np.array([
        [1, 2, 3, 0],
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [0, 1, 2, 3]
    ])
    kernel_2 = np.array([
        [1, 0],
        [0, -1]
    ])
    bias_2 = 1.0
    stride_2 = 2
    padding_2 = 0

    # Corrected expected output after recalculations
    expected_output_2 = np.array([
        [ 1., 1.],
        [ 3., -1.]
    ])
    output_2 = cnn_forward(input_matrix_2, kernel_2, bias_2, stride_2, padding_2)
    assert np.array_equal(output_2, expected_output_2), f"Test case 2 failed: Expected output {expected_output_2}, got {output_2}"

    # Test case 3: Different kernel and padding
    input_matrix_3 = np.array([
        [0, 1, 2],
        [2, 3, 0],
        [1, 0, 1]
    ])
    kernel_3 = np.array([
        [2, 1],
        [0, -1]
    ])
    bias_3 = 0.5
    stride_3 = 1
    padding_3 = 1

    # Corrected expected output after recalculations
    expected_output_3 = np.array([
        [ 0.5, -0.5, -1.5,  0.5],
        [ -1.5,  -1.5,  4.5,  4.5],
        [ 1.5,  7.5,  5.5,  0.5],
        [ 1.5,  2.5,  1.5,  2.5]
    ])
    output_3 = cnn_forward(input_matrix_3, kernel_3, bias_3, stride_3, padding_3)
    assert np.array_equal(output_3, expected_output_3), f"Test case 3 failed: Expected output {expected_output_3}, got {output_3}"

    print("Test case 1 output:")
    print(output_1)
    print("Test case 2 output:")
    print(output_2)
    print("Test case 3 output:")
    print(output_3)

if __name__ == "__main__":
    test_cnn_forward()
    print("All CNN tests passed")