import torch
from typing import List, Tuple, Union


def train_neuron(
    features: Union[List[List[float]], torch.Tensor],
    labels: Union[List[float], torch.Tensor],
    initial_weights: Union[List[float], torch.Tensor],
    initial_bias: float,
    learning_rate: float,
    epochs: int,
) -> Tuple[List[float], float, List[float]]:
    # Ensure tensors
    X = torch.as_tensor(features, dtype=torch.float)
    y = torch.as_tensor(labels, dtype=torch.float)
    w = torch.as_tensor(initial_weights, dtype=torch.float)
    b = torch.tensor(initial_bias, dtype=torch.float)

    n = y.shape[0]
    mse_values: List[float] = []

    for _ in range(epochs):
        # Forward
        z = X @ w + b  # (n,)
        preds = torch.sigmoid(z)  # (n,)
        errors = preds - y  # (n,)

        # MSE
        mse = torch.mean(errors**2).item()
        mse_values.append(round(mse, 4))

        # Manual gradients (chain-rule): dMSE/dz = 2/n * (preds-y) * σ'(z)
        sigma_prime = preds * (1 - preds)
        delta = (2.0 / n) * errors * sigma_prime  # (n,)

        grad_w = X.t() @ delta  # (f,)
        grad_b = delta.sum()  # scalar

        # Parameter update (gradient descent)
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

    # Round final params for return
    updated_weights = [round(val, 4) for val in w.tolist()]
    updated_bias = round(b.item(), 4)
    return updated_weights, updated_bias, mse_values
