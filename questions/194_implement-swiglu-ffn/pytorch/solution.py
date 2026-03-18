import torch
import torch.nn.functional as F

def swiglu_ffn(x, W1, W3, W2):
    gate = F.silu(x @ W1.T)
    up = x @ W3.T
    return (gate * up) @ W2.T
