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
    """
    Train a single neuron (sigmoid activation) with mean-squared-error loss.

    Returns (updated_weights, updated_bias, mse_per_epoch)
    — weights & bias are rounded to 4 decimals; each MSE value is rounded too.
    """
    # Your implementation here
    pass
