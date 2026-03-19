import numpy as np

def lightning_attention(Q, K, V, decay):
    seq_len, head_dim = Q.shape
    S = np.zeros((head_dim, head_dim))
    output = np.zeros((seq_len, head_dim))

    for t in range(seq_len):
        S = decay * S + np.outer(K[t], V[t])
        output[t] = Q[t] @ S

    return output.astype(float)
