import numpy as np

def lstm_forward(input_sequence, initial_hidden_state, initial_cell_state, Wf, Wi, Wc, Wo, bf, bi, bc, bo):
    """
    Computes the forward pass of a simple LSTM using numpy arrays.

    Parameters:
        input_sequence (numpy array): The input sequence, where each row represents an input at time t.
        initial_hidden_state (numpy array): The initial hidden state of the LSTM.
        initial_cell_state (numpy array): The initial cell state of the LSTM.
        Wf, Wi, Wc, Wo (numpy array): Weight matrices for the forget gate, input gate, cell candidate, and output gate, respectively.
        bf, bi, bc, bo (numpy array): Bias vectors for the forget gate, input gate, cell candidate, and output gate, respectively.

    Returns:
        numpy array: The final hidden state after processing the input sequence.
        numpy array: The final cell state.
    """
    h_prev = initial_hidden_state
    c_prev = initial_cell_state

    for x_t in input_sequence:
        # Concatenate input and previous hidden state
        combined = np.concatenate((x_t, h_prev))
        # Forget gate
        f_t = sigmoid(np.dot(Wf, combined) + bf)

        # Input gate
        i_t = sigmoid(np.dot(Wi, combined) + bi)

        # Cell candidate
        c_hat_t = np.tanh(np.dot(Wc, combined) + bc)

        # Update cell state
        c_t = f_t * c_prev + i_t * c_hat_t

        # Output gate
        o_t = sigmoid(np.dot(Wo, combined) + bo)

        # Update hidden state
        h_t = o_t * np.tanh(c_t)

        # Update previous hidden and cell states for next time step
        h_prev = h_t
        c_prev = c_t

    return h_t, c_t

def sigmoid(x):
    """ Sigmoid activation function using numpy """
    return 1 / (1 + np.exp(-x))

def test_lstm_forward():
    # Test case 1: Single feature inputs
    input_sequence_1 = np.array([[1.0], [2.0], [3.0]])
    initial_hidden_state_1 = np.array([0.0])
    initial_cell_state_1 = np.array([0.0])
    Wf_1 = np.array([[0.1, 0.1]])  # Adjusted to have 2 columns instead of 1
    Wi_1 = np.array([[0.2, 0.2]])  # Similarly adjust all weight matrices
    Wc_1 = np.array([[0.3, 0.3]])
    Wo_1 = np.array([[0.4, 0.4]])

    bf_1 = np.array([0.1])
    bi_1 = np.array([0.2])
    bc_1 = np.array([0.3])
    bo_1 = np.array([0.4])

    final_hidden_state_1, final_cell_state_1 = lstm_forward(input_sequence_1, initial_hidden_state_1, initial_cell_state_1, Wf_1, Wi_1, Wc_1, Wo_1, bf_1, bi_1, bc_1, bo_1)
    expected_hidden_state_1 = np.array([0.66260894])  # Expected value needs manual calculation or pre-calculated expectation
    expected_cell_state_1 = np.array([1.0298676])

    assert np.allclose(final_hidden_state_1, expected_hidden_state_1, atol=0.01), f"Test case 1 failed: expected {expected_hidden_state_1}, got {final_hidden_state_1}"
    assert np.allclose(final_cell_state_1, expected_cell_state_1, atol=0.01), f"Test case 1 failed: expected {expected_cell_state_1}, got {final_cell_state_1}"

    # Test case 2: Single feature inputs with different weights and bias
    input_sequence_2 = np.array([[0.5], [0.1], [-0.2]])
    initial_hidden_state_2 = np.array([0.0])
    initial_cell_state_2 = np.array([0.0])
    Wf_2 = np.array([[0.5, 0.5]])  # Adjusted to have 2 columns instead of 1
    Wi_2 = np.array([[0.2, 0.2]])  # Similarly adjust all weight matrices
    Wc_2 = np.array([[0.3, 0.3]])
    Wo_2 = np.array([[0.4, 0.4]])
    bf_2 = np.array([0.2])
    bi_2 = np.array([0.1])
    bc_2 = np.array([0.5])
    bo_2 = np.array([0.3])

    final_hidden_state_2, final_cell_state_2 = lstm_forward(input_sequence_2, initial_hidden_state_2, initial_cell_state_2, Wf_2, Wi_2, Wc_2, Wo_2, bf_2, bi_2, bc_2, bo_2)
    expected_hidden_state_2 = np.array([0.27429882])  # Expected value needs manual calculation or pre-calculated expectation
    expected_cell_state_2 = np.array([0.51318488])

    assert np.allclose(final_hidden_state_2, expected_hidden_state_2, atol=0.01), f"Test case 2 failed: expected {expected_hidden_state_2}, got {final_hidden_state_2}"
    assert np.allclose(final_cell_state_2, expected_cell_state_2, atol=0.01), f"Test case 2 failed: expected {expected_cell_state_2}, got {final_cell_state_2}"

    print("All LSTM tests passed.")

if __name__ == "__main__":
    test_lstm_forward()
