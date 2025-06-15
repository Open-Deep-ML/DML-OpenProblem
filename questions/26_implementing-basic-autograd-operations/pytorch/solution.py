import torch


class Value:
    """Scalar autograd value powered by PyTorch tensors."""

    def __init__(self, data, _tensor=None):
        self._t = (
            _tensor
            if _tensor is not None
            else torch.tensor(float(data), requires_grad=True)
        )
        self._t.retain_grad()

    # ----- helpers -----
    @property
    def data(self):
        return self._t.item()

    @property
    def grad(self):
        g = self._t.grad
        return 0 if g is None else g.item()

    def __repr__(self):
        def fmt(x):
            return int(x) if float(x).is_integer() else round(x, 4)

        return f"Value(data={fmt(self.data)}, grad={fmt(self.grad)})"

    def _wrap(self, other):
        return other if isinstance(other, Value) else Value(other)

    # arithmetic
    def __add__(self, other):
        other = self._wrap(other)
        return Value(0, _tensor=self._t + other._t)

    __radd__ = __add__

    def __mul__(self, other):
        other = self._wrap(other)
        return Value(0, _tensor=self._t * other._t)

    __rmul__ = __mul__

    # activation
    def relu(self):
        return Value(0, _tensor=torch.relu(self._t))

    # back-prop
    def backward(self):
        self._t.backward()
