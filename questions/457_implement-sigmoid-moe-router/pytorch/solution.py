import torch

def sigmoid_moe_router(hidden_states, gate_weight, score_bias, top_k):
    logits = hidden_states @ gate_weight.T
    routing_weights = torch.sigmoid(logits)
    scores_for_choice = routing_weights + score_bias

    top_k_indices = torch.topk(scores_for_choice, top_k, dim=-1).indices

    top_k_weights = torch.gather(routing_weights, 1, top_k_indices)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

    return top_k_weights.float(), top_k_indices
