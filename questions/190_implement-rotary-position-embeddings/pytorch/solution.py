import torch

def apply_rotary_embeddings(q, k, positions, dim, base=10000.0):
    seq_len, head_dim = q.shape

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=q.device) / dim))
    angles = torch.outer(positions.float().to(q.device), freqs)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    def rotate(x):
        x_rot = x[:, :dim]
        x_pass = x[:, dim:]
        x1 = x_rot[:, :dim // 2]
        x2 = x_rot[:, dim // 2:]
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return torch.cat([rotated, x_pass], dim=-1).float()

    return rotate(q), rotate(k)
