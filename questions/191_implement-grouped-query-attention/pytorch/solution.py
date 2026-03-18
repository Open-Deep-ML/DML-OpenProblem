import torch
import torch.nn.functional as F

def grouped_query_attention(Q, K, V, num_kv_heads):
    num_q_heads, seq_len, head_dim = Q.shape
    group_size = num_q_heads // num_kv_heads

    K_expanded = K.repeat_interleave(group_size, dim=0)
    V_expanded = V.repeat_interleave(group_size, dim=0)

    scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, V_expanded)
    return output.float()
