def polynomial_derivative(x: float,n: float) -> float:
    if n == 0.0:
        return 0.0
    return round(n*(x**(n-1)),4)

def test_polynomial_derivative():
    # Test case 1
    x,n = 10.0,0.0
    expected_output = 0.0
    assert polynomial_derivative(x,n) == expected_output, "Test case 1 failed"

    # Test case 2
    x,n = 6.0,5.0
    expected_output = 6480.0
    assert polynomial_derivative(x,n) == expected_output, "Test case 2 failed"

    # Test case 3
    x,n = 12.0,4.3
    expected_output = 15659.0917
    assert polynomial_derivative(x,n) == expected_output, "Test case 3 failed"

    # Test case 4
    x,n = 14.35,1.0
    expected_output = 1.0
    assert polynomial_derivative(x,n) == expected_output, "Test case 4 failed"

if __name__ == "__main__":
    test_polynomial_derivative()
    print("All Polynomial Derivative tests passed.")