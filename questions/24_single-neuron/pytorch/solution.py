import torch
import torch.nn.functional as F
from typing import List, Tuple


def single_neuron_model(
    features: List[List[float]], labels: List[float], weights: List[float], bias: float
) -> Tuple[List[float], float]:
    X = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.float)
    w = torch.tensor(weights, dtype=torch.float)
    b = torch.tensor(bias, dtype=torch.float)

    logits = X.matmul(w) + b
    probs_t = torch.sigmoid(logits)
    probs = probs_t.tolist()

    mse = F.mse_loss(probs_t, y, reduction="mean").item()

    return probs, mse
