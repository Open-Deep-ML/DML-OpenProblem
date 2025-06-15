def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
    if len(a) != len(b):
        return -1
    return [a[i] + b[i] for i in range(len(a))]
