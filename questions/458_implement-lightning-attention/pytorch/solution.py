import torch

def lightning_attention(Q, K, V, decay):
    seq_len, head_dim = Q.shape
    S = torch.zeros((head_dim, head_dim), dtype=Q.dtype, device=Q.device)
    output = torch.zeros_like(Q)

    for t in range(seq_len):
        S = decay * S + torch.outer(K[t], V[t])
        output[t] = Q[t] @ S

    return output.float()
