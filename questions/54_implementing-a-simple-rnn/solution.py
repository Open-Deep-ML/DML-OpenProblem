
import numpy as np

def rnn_forward(input_sequence, initial_hidden_state, Wx, Wh, b):
    h = np.array(initial_hidden_state)
    Wx = np.array(Wx)
    Wh = np.array(Wh)
    b = np.array(b)
    for x in input_sequence:
        x = np.array(x)
        h = np.tanh(np.dot(Wx, x) + np.dot(Wh, h) + b)
    final_hidden_state = np.round(h, 4)
    return final_hidden_state.tolist()
