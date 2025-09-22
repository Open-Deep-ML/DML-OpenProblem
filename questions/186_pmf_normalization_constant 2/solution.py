import math

def find_k():
    """
    Solve 10*K^2 + 9*K - 1 = 0 and return the non-negative root.
    """
    a = 10.0
    b = 9.0
    c = -1.0
    discriminant = b * b - 4 * a * c
    sqrt_disc = math.sqrt(discriminant)
    k1 = (-b + sqrt_disc) / (2 * a)
    k2 = (-b - sqrt_disc) / (2 * a)
    return k1 if k1 >= 0 else k2
