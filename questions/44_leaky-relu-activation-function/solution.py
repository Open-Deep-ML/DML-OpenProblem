def leaky_relu(z: float, alpha: float = 0.01) -> float | int:
    return z if z > 0 else alpha * z
