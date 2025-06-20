import numpy as np


def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    if len(p) != len(q):
        return 0.0

    p, q = np.array(p), np.array(q)
    BC = np.sum(np.sqrt(p * q))
    DB = -np.log(BC)
    return round(DB, 4)
