import torch


def linear_regression_normal_equation(X, y) -> torch.Tensor:
    """
    Solve linear regression via the normal equation using PyTorch.
    X: Tensor or convertible of shape (m,n); y: shape (m,) or (m,1).
    Returns a 1-D tensor of length n, rounded to 4 decimals.
    """
    X_t = torch.as_tensor(X, dtype=torch.float)
    y_t = torch.as_tensor(y, dtype=torch.float).reshape(-1, 1)
    # normal equation: theta = (XᵀX)⁻¹ Xᵀ y
    XtX = X_t.transpose(0, 1) @ X_t
    theta = torch.linalg.inv(XtX) @ (X_t.transpose(0, 1) @ y_t)
    theta = theta.flatten()
    return torch.round(theta * 10000) / 10000
