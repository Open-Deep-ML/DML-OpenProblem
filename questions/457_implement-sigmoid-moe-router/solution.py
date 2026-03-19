import numpy as np

def sigmoid_moe_router(hidden_states, gate_weight, score_bias, top_k):
    logits = hidden_states @ gate_weight.T
    routing_weights = 1.0 / (1.0 + np.exp(-logits))
    scores_for_choice = routing_weights + score_bias

    top_k_indices = np.argsort(-scores_for_choice, axis=-1)[:, :top_k]

    top_k_weights = np.take_along_axis(routing_weights, top_k_indices, axis=-1)
    top_k_weights = top_k_weights / np.sum(top_k_weights, axis=-1, keepdims=True)

    return top_k_weights.astype(float), top_k_indices
