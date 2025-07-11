import numpy as np

def ucb_action(counts, values, t, c):
    counts = np.array(counts)
    values = np.array(values)
    ucb = values + c * np.sqrt(np.log(t) / (counts + 1e-8))
    return int(np.argmax(ucb))
