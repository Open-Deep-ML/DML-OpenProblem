import numpy as np
from scipy.special import softmax


def get_top_k(arr: np.ndarray, k: int):
    """
    Use this function to get the top k values and their indices along last dimension.

    Example
    ---
    >>> a = np.arange(8).reshape(2, 4)
    >>> idx, vals = get_top_k(a, 2)

    a: [[0, 1, 2, 3],
        [4, 5, 6, 7]]
    idx: [[2, 3], [2, 3]] <- last two indices in each row
    vals: [[2, 3], [6, 7]] <- last two values in each row

    **This function should be provided to the user**
    """
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

    # flatten batch and sequence dimentions for easier indexing
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

    # prepare result memeory for aggregation: [n_tokens, d_model]
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


def test_moe():
    d_model = 2
    n_experts = 4
    l_seq = 3
    n_batch = 2
    top_k = 2

    np.random.seed(42)
    x = np.random.rand(n_batch, l_seq, d_model)
    We = np.random.rand(n_experts, d_model, d_model)
    Wg = np.random.rand(d_model, n_experts)

    output = moe(x, We, Wg, n_experts, top_k)
    np.testing.assert_allclose(
        output,
        np.array(
            [
                [
                    [0.51476239, 0.43288392],
                    [0.55535899, 0.54474649],
                    [0.12853074, 0.10201816],
                ],
                [
                    [0.33898452, 0.3045885],
                    [0.5391176, 0.41704218],
                    [0.35968146, 0.32618981],
                ],
            ]
        ),
    )

    d_model = 2
    n_experts = 4
    l_seq = 3
    n_batch = 2
    top_k = 2

    np.random.seed(42)
    x = np.random.rand(n_batch, l_seq, d_model)
    We = np.zeros((n_experts, d_model, d_model))
    Wg = np.random.rand(d_model, n_experts)

    output = moe(x, We, Wg, n_experts, top_k)
    np.testing.assert_allclose(
        output,
        np.array(
            [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]
        ),
    )

    d_model = 2
    n_experts = 4
    l_seq = 3
    n_batch = 2
    top_k = 1

    np.random.seed(42)
    x = np.arange(12).reshape(n_batch, l_seq, d_model)
    We = np.ones((n_experts, d_model, d_model))
    Wg = np.ones((d_model, n_experts))

    output = moe(x, We, Wg, n_experts, top_k)
    np.testing.assert_allclose(
        output, np.array([[[1, 1], [5, 5], [9, 9]], [[13, 13], [17, 17], [21, 21]]])
    )


if __name__ == "__main__":
    test_moe()
    print("All MoE test cases passed.")
