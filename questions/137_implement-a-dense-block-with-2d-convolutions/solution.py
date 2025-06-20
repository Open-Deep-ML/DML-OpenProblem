import numpy as np


def conv2d(x, kernel, padding=0):
    if padding > 0:
        x_padded = np.pad(
            x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode="constant"
        )
    else:
        x_padded = x
    batch_size, in_height, in_width, in_channels = x_padded.shape
    kh, kw, _, out_channels = kernel.shape
    out_height = in_height - kh + 1
    out_width = in_width - kw + 1
    output = np.zeros((batch_size, out_height, out_width, out_channels))
    for b in range(batch_size):
        for i in range(out_height):
            for j in range(out_width):
                for c_out in range(out_channels):
                    sum_val = 0.0
                    for c_in in range(in_channels):
                        sum_val += np.sum(
                            x_padded[b, i : i + kh, j : j + kw, c_in]
                            * kernel[:, :, c_in, c_out]
                        )
                    output[b, i, j, c_out] = sum_val
    return output


def dense_net_block(input_data, num_layers, growth_rate, kernels, kernel_size=(3, 3)):
    kh, kw = kernel_size
    padding = (kh - 1) // 2
    concatenated_features = input_data.copy()
    for l in range(num_layers):
        activated = np.maximum(concatenated_features, 0.0)
        conv_output = conv2d(activated, kernels[l], padding=padding)
        concatenated_features = np.concatenate(
            [concatenated_features, conv_output], axis=3
        )
    return concatenated_features
