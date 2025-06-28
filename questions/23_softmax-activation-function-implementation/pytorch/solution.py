import torch
import torch.nn.functional as F


def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities,
        each rounded to 4 decimals.
    """
    scores_t = torch.as_tensor(scores, dtype=torch.float)
    probs = F.softmax(scores_t, dim=0)
    probs = torch.round(probs * 10000) / 10000
    return probs.tolist()
