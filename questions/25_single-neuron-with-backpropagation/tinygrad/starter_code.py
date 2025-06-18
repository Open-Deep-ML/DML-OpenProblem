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
    """
    Tinygrad version — same contract as PyTorch implementation.
    """
    # Your implementation here
    pass
