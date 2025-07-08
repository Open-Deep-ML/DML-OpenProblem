import torch

def SwiGLU(x: torch.Tensor) -> torch.Tensor:
    """
    Apply the SwiGLU activation function.
    Assumes:
      - Input x is a torch tensor of shape (batch_size, 2d)
      - x has already been passed through a linear projection layer

    Returns:
      - Tensor of shape (batch_size, d) after applying SwiGLU:
        x1 * SiLU(x2), where [x1, x2] = split(x)
    """
    x1, x2 = x.chunk(2, dim=-1)
    return x1 * torch.nn.functional.silu(x2)
