## Problem

Write a Python function `sigmoid_moe_router(hidden_states, gate_weight, score_bias, top_k)` that implements the sigmoid-based Mixture of Experts routing used in MiniMax M2.5. The function takes:
- `hidden_states`: array of shape `(num_tokens, hidden_dim)`
- `gate_weight`: array of shape `(num_experts, hidden_dim)`
- `score_bias`: array of shape `(num_experts,)` for expert load balancing
- `top_k`: number of experts to select per token

The function should:
1. Compute router logits via matrix multiplication
2. Apply sigmoid activation (not softmax) to get routing weights
3. Add the score bias to determine expert selection
4. Select the top-k experts per token
5. Gather the actual sigmoid weights (without bias) for selected experts
6. Normalize the selected weights to sum to 1

Return a tuple of `(top_k_weights, top_k_indices)` where weights has shape `(num_tokens, top_k)` and indices has shape `(num_tokens, top_k)`. Only use numpy.
