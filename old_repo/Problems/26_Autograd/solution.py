class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


def test_value_operations():
    # Test case setup
    a = Value(2)
    b = Value(3)
    c = Value(10)
    d = a + b * c
    e = Value(7) * Value(2)
    f = e + d
    g = f.relu()
    g.backward()

    # Manually set the expected `grad` values for each object
    expected = [
        (2, 1),  # a
        (3, 10),  # b
        (10, 3),  # c
        (32, 1),  # d
        (14, 1),  # e
        (46, 1),  # f
        (46, 1),  # g
    ]

    # Actual results
    results = [a, b, c, d, e, f, g]
    for i, (result, (expected_data, expected_grad)) in enumerate(zip(results, expected)):
        assert result.data == expected_data, f"Test failed at step {i}: data mismatch ({result.data} != {expected_data})"
        assert result.grad == expected_grad, f"Test failed at step {i}: grad mismatch ({result.grad} != {expected_grad})"

if __name__ == "__main__":
    test_value_operations()
    print("All Value operation tests passed.")
