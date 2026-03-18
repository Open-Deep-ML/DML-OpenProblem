import numpy as np

def apply_rotary_embeddings(q, k, positions, dim, base=10000.0):
    seq_len, head_dim = q.shape

    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=float) / dim))
    angles = np.outer(positions.astype(float), freqs)
    cos = np.cos(angles)
    sin = np.sin(angles)

    def rotate(x):
        x_rot = x[:, :dim]
        x_pass = x[:, dim:]
        x1 = x_rot[:, :dim // 2]
        x2 = x_rot[:, dim // 2:]
        rotated = np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
        return np.concatenate([rotated, x_pass], axis=-1).astype(float)

    return rotate(q), rotate(k)
