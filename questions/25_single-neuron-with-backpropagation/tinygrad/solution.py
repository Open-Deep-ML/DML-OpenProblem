from tinygrad.tensor import Tensor
from typing import List, Tuple


def train_neuron_tg(
    features: List[List[float]],
    labels:   List[float],
    initial_weights: List[float],
    initial_bias: float,
    learning_rate: float,
    epochs: int
) -> Tuple[List[float], float, List[float]]:
    X = Tensor(features)
    y = Tensor(labels).reshape(len(labels), 1)
    w = Tensor(initial_weights).reshape(len(initial_weights), 1)
    b = Tensor(initial_bias)

    mse_values: List[float] = []
    n = len(labels)

    for _ in range(epochs):
        z = X.matmul(w) + b         # (n,1)
        preds = z.sigmoid()         # (n,1)
        errors = preds - y          # (n,1)

        mse = float(((errors**2).mean()).numpy())
        mse_values.append(round(mse, 4))

        sigma_prime = preds * (1 - preds)
        delta = (2.0 / n) * errors * sigma_prime  # (n,1)

        grad_w = X.T.matmul(delta)  # (f,1)
        grad_b = delta.sum()

        w -= Tensor(learning_rate) * grad_w
        b -= Tensor(learning_rate) * grad_b

    updated_weights = [round(val, 4) for val in w.numpy().flatten().tolist()]
    updated_bias    = round(float(b.numpy()), 4)
    return updated_weights, updated_bias, mse_values
