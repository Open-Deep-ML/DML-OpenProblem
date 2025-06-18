import torch

def determinant_4x4(matrix) -> float:
    """
    Compute the determinant of a 4×4 matrix using PyTorch.
    Input can be a Python list, NumPy array, or torch Tensor of shape (4,4).
    Returns a Python float.
    """
    # Convert to tensor
    m = torch.as_tensor(matrix, dtype=torch.float)
    # Your implementation here
    pass
