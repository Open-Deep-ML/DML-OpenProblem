from tinygrad.tensor import Tensor
from typing import List, Tuple

def single_neuron_model_tg(
    features: List[List[float]],
    labels: List[float],
    weights: List[float],
    bias: float
) -> Tuple[List[float], float]:
    X = Tensor(features)
    w = Tensor(weights)
    b = bias
    probs: List[float] = []
    for i in range(len(features)):
        z = X[i].dot(w) + b
        p = z.sigmoid().numpy().tolist()
        probs.append(round(p, 4))

    mse = sum((p - y)**2 for p, y in zip(probs, labels)) / len(labels)
    mse = round(mse, 4)

    return probs, mse
