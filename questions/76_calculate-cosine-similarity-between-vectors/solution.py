import numpy as np


def cosine_similarity(v1, v2):
    if v1.shape != v2.shape:
        raise ValueError("Arrays must have the same shape")

    if v1.size == 0:
        raise ValueError("Arrays cannot be empty")

    # Flatten arrays in case of 2D
    v1_flat = v1.flatten()
    v2_flat = v2.flatten()

    dot_product = np.dot(v1_flat, v2_flat)
    magnitude1 = np.sqrt(np.sum(v1_flat**2))
    magnitude2 = np.sqrt(np.sum(v2_flat**2))

    if magnitude1 == 0 or magnitude2 == 0:
        raise ValueError("Vectors cannot have zero magnitude")

    return round(dot_product / (magnitude1 * magnitude2), 3)
