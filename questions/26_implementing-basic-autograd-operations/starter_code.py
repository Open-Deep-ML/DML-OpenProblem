class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        # Implement addition here
        pass

    def __mul__(self, other):
        # Implement multiplication here
        pass

    def relu(self):
        # Implement ReLU here
        pass

    def backward(self):
        # Implement backward pass here
        pass
