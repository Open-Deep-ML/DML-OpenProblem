import torch

def sigmoid_moe_router(hidden_states, gate_weight, score_bias, top_k):
    """
    Implement sigmoid-based MoE routing with bias correction.

    Args:
        hidden_states (torch.Tensor): Token representations, shape (num_tokens, hidden_dim).
        gate_weight (torch.Tensor): Gate projection weights, shape (num_experts, hidden_dim).
        score_bias (torch.Tensor): Learned bias for load balancing, shape (num_experts,).
        top_k (int): Number of experts to select per token.

    Returns:
        tuple: (top_k_weights, top_k_indices)
    """
    pass
