import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def get_top_k(arr: np.ndarray, k: int):
    idx = np.argpartition(arr, -k)[..., -k:]
    vals = np.take_along_axis(arr, idx, axis=-1)
    return idx, vals


def expert(x: np.ndarray, We_i: np.ndarray):
    # x: [n_tokens, d_model]
    # We_i: [d_model, d_model]
    return x @ We_i


def gate(x: np.ndarray, Wg: np.ndarray):
    # x: [n_batch * l_seq, d_model]
    # Wg: [n_batch * l_seq, n_experts]
    return x @ Wg


def moe(x: np.ndarray, We: np.ndarray, Wg: np.ndarray, n_experts: int, top_k: int):
    # x: [n_batch, l_seq, d_model]
    # We: [n_experts, d_model, d_model]
    # Wg: [n_batch * l_seq, n_experts]

    n_batch, l_seq, d_model = x.shape

    # flatten batch and sequence dimensions for easier indexing
    # x_flat: [n_batch * l_seq, d_model]
    x_flat = x.reshape(-1, d_model)
    n_tokens, _ = x_flat.shape

    gating_logits = gate(x_flat, Wg)
    gating_weights = softmax(gating_logits, axis=-1)

    topk_idx, topk_weights = get_top_k(gating_weights, top_k)
    topk_idx_flat = topk_idx.flatten()  # [n_tokens * top_k]
    # mapping from top K expert indices to token indices: [n_tokens * top_k]
    token_idx_flat = np.arange(n_tokens).repeat(top_k)

    topk_weights_norm = topk_weights / topk_weights.sum(axis=1, keepdims=True)
    topk_weights_norm_flat = topk_weights_norm.flatten()

    # prepare result memory for aggregation: [n_tokens, d_model]
    output_flat = np.zeros_like(x_flat)
    for i in range(n_experts):
        mask = topk_idx_flat == i
        tokens_expert_i = token_idx_flat[mask]

        if tokens_expert_i.size > 0:
            x_expert_i = x_flat[tokens_expert_i]
            output_expert_i = expert(x_expert_i, We[i, ...])
            output_expert_i *= topk_weights_norm_flat[mask, None]

            # scatter add to result memory
            np.add.at(output_flat, tokens_expert_i, output_expert_i)

    return output_flat.reshape(n_batch, l_seq, d_model)
