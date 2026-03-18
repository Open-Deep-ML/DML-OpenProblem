import numpy as np

def grouped_query_attention(Q, K, V, num_kv_heads):
    num_q_heads, seq_len, head_dim = Q.shape
    group_size = num_q_heads // num_kv_heads

    K_expanded = np.repeat(K, group_size, axis=0)
    V_expanded = np.repeat(V, group_size, axis=0)

    scores = np.matmul(Q, K_expanded.transpose(0, 2, 1)) / np.sqrt(head_dim)
    attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)

    output = np.matmul(attn_weights, V_expanded)
    return output.astype(float)
