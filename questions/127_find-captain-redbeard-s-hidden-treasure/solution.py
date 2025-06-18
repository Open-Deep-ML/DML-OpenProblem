def find_treasure(start_x: float) -> float:
	learning_rate = 0.1, tolerance = 1e-6, max_iters = 10000)    def gradient(x):
        return 4 * x**3 - 9 * x**2  # derivative of x^4 - 3x^3 + 2

    x = start_x
    for _ in range(max_iters):
        grad = gradient(x)
        new_x = x - learning_rate * grad
        if abs(new_x - x) < tolerance:
            break
        x = new_x
    return round(x, 4)
