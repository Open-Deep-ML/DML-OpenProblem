import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
    h = np.array(initial_hidden_state)
    Wx = np.array(Wx)
    Wh = np.array(Wh)
    b = np.array(b)
    for x in input_sequence:
        x = np.array(x)
        h = np.tanh(np.dot(Wx, x) + np.dot(Wh, h) + b)
    final_hidden_state = h
    return final_hidden_state

def test_rnn_forward():
    # Test case 1: Single feature inputs
    input_sequence_1 = [[1.0], [2.0], [3.0]]
    initial_hidden_state_1 = [0.0]
    Wx_1 = [[0.5]]  # Input to hidden weights
    Wh_1 = [[0.8]]  # Hidden to hidden weights
    b_1 = [0.0]      # Bias
    expected_output_1 = [0.97588162]
    output_1 = rnn_forward(input_sequence_1, initial_hidden_state_1, Wx_1, Wh_1, b_1)
    assert np.allclose(output_1, expected_output_1, atol=0.01), f"Test case 1 failed: expected {expected_output_1}, got {output_1}"

    # Test case 2: Single feature inputs with different weights and bias
    input_sequence_2 = [[0.5], [0.1], [-0.2]]
    initial_hidden_state_2 = [0.0]
    Wx_2 = [[1.0]]  # Input to hidden weights
    Wh_2 = [[0.5]]   # Hidden to hidden weights
    b_2 = [0.1]      # Bias
    expected_output_2 = [0.118]
    output_2 = rnn_forward(input_sequence_2, initial_hidden_state_2, Wx_2, Wh_2, b_2)
    assert np.allclose(output_2, expected_output_2, atol=0.01), f"Test case 2 failed: expected {expected_output_2}, got {output_2}"

    # Test case 3: Multiple feature inputs
    input_sequence_3 = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]
    initial_hidden_state_3 = [0.0, 0.0]
    Wx_3 = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]  # Input to hidden weights
    Wh_3 = [
        [0.7, 0.8],
        [0.9, 1.0]
    ]  # Hidden to hidden weights
    b_3 = [0.1, 0.2]  # Bias
    expected_output_3 = [0.7474, 0.9302]
    output_3 = rnn_forward(input_sequence_3, initial_hidden_state_3, Wx_3, Wh_3, b_3)
    assert np.allclose(output_3, expected_output_3, atol=0.01), f"Test case 3 failed: expected {expected_output_3}, got {output_3}"

if __name__ == "__main__":
    test_rnn_forward()
    print("All RNN tests passed.")
