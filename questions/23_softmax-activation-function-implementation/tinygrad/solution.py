from tinygrad.tensor import Tensor

def softmax_tg(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using tinygrad.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities,
        each rounded to 4 decimals.
    """
    t = Tensor(scores)
    t_max = t.max()
    exp_scores = (t - t_max).exp()
    probs = exp_scores / exp_scores.sum()
    probs_list = probs.numpy().tolist()
    return [round(p, 4) for p in probs_list]
