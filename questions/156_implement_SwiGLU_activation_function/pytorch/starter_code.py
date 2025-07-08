import torch

def swiglu(x: torch.Tensor) -> torch.Tensor:
    """
    Apply the SwiGLU activation function.
    Assumes:
      - Input x is a torch tensor of shape (batch_size, 2d)
      - x has already been passed through a linear projection layer

    Returns:
      - Tensor of shape (batch_size, d) after applying SwiGLU:
    """
    return scores
