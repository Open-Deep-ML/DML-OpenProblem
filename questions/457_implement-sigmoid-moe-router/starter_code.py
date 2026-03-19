import numpy as np

# Implement your function below.
def sigmoid_moe_router(hidden_states, gate_weight, score_bias, top_k):
    """
    Implement sigmoid-based MoE routing with bias correction.

    Args:
        hidden_states (np.ndarray): Token representations, shape (num_tokens, hidden_dim).
        gate_weight (np.ndarray): Gate projection weights, shape (num_experts, hidden_dim).
        score_bias (np.ndarray): Learned bias for load balancing, shape (num_experts,).
        top_k (int): Number of experts to select per token.

    Returns:
        tuple: (top_k_weights, top_k_indices)
            - top_k_weights: Normalized routing weights, shape (num_tokens, top_k).
            - top_k_indices: Selected expert indices, shape (num_tokens, top_k).
    """
    pass
