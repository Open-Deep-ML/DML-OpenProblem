import torch

def log_softmax_stable(x: torch.Tensor) -> torch.Tensor:
    x_max = x.max(dim=-1, keepdim=True).values
    log_probs = x - x_max - torch.log(torch.exp(x - x_max).sum(dim=-1, keepdim=True))
    return log_probs

def cross_entropy_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    log_probs = log_softmax_stable(y_pred)
    loss = -log_probs[range(len(y_true)), y_true].mean()
    return loss