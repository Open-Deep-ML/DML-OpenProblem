import numpy as np

def swiglu_ffn(x, W1, W3, W2):
    gate = x @ W1.T
    silu_gate = gate * (1.0 / (1.0 + np.exp(-gate)))
    up = x @ W3.T
    hidden = silu_gate * up
    output = hidden @ W2.T
    return output.astype(float)
