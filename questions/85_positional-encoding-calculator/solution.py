import numpy as np


def pos_encoding(position: int, d_model: int):
    if position == 0 or d_model <= 0:
        return -1

    # Create position and dimension indices
    pos = np.arange(position, dtype=np.float32).reshape(position, 1)
    ind = np.arange(d_model, dtype=np.float32).reshape(1, d_model)

    # Compute the angles
    angle_rads = pos / np.power(10000, (2 * (ind // 2)) / d_model)

    # Apply sine to even indices, cosine to odd indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # Even indices (0, 2, 4...)
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # Odd indices (1, 3, 5...)

    # Convert to float16 as required
    return angle_rads.astype(np.float16)
