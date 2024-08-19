def leaky_relu(z: float, alpha: float = 0.01) -> float:
    return z if z > 0 else alpha * z


def test_leaky_relu():
    # Test case 1: z = 0
    assert leaky_relu(0) == 0, "Test case 1 failed"

    # Test case 2: z = 1
    assert leaky_relu(1) == 1, "Test case 2 failed"

    # Test case 3: z = -1
    assert leaky_relu(-1) == -0.01, "Test case 3 failed"

    # Test case 4: z = -2, alpha = 0.1
    assert leaky_relu(-2, alpha=0.1) == -0.2, "Test case 4 failed"


if __name__ == "__main__":
    test_leaky_relu()
    print("All Leaky ReLU tests passed.")
