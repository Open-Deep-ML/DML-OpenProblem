import numpy as np

def noisy_topk_gating(
    X: np.ndarray,
    W_g: np.ndarray,
    W_noise: np.ndarray,
    N: np.ndarray,
    k: int
) -> np.ndarray:
    H_base = X @ W_g
    H_noise = X @ W_noise
    softplus = np.log1p(np.exp(H_noise))
    H = H_base + N * softplus

    def top_k_masked(row, k):
        mask = np.full_like(row, -np.inf)
        top_idx = np.argsort(row)[-k:]
        mask[top_idx] = row[top_idx]
        return mask

    masked_H = np.vstack([top_k_masked(row, k) for row in H])
    exps = np.exp(masked_H - np.max(masked_H, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)
