import torch
import torch.nn as nn

class DropoutLayer(nn.Module):
    def __init__(self, p: float):
        """Initialize the dropout layer."""
        super(DropoutLayer, self).__init__()
        if p < 0 or p >= 1:
            raise ValueError("Dropout rate must be between 0 and 1 (1-exclusive)")
        
        self.p = p
        self.mask = None
        
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Forward pass of the dropout layer."""
        if not training:
            return x
                
        self.mask = torch.bernoulli(torch.ones_like(x) * (1 - self.p))
        
        return x * self.mask / (1 - self.p)
        
    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        """Backward pass of the dropout layer."""
        if self.mask is None:
            raise ValueError("Forward pass must be called before backward pass")
            
        return grad * self.mask / (1 - self.p)
