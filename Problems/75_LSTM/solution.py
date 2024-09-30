import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)
        
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        h = initial_hidden_state
        c = initial_cell_state
        outputs = []

        for t in range(len(x)):
            xt = x[t].reshape(-1, 1)
            concat = np.vstack((h, xt))

            # Forget gate
            ft = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
            
            # Input gate
            it = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
            
            # Candidate cell state
            c_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)
            
            # Cell state update
            c = ft * c + it * c_tilde
            
            # Output gate
            ot = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
            
            # Hidden state update
            h = ot * np.tanh(c)
            
            outputs.append(h)

        return np.array(outputs), h, c

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

def test_lstm():
    # Test case 1: Single feature inputs
    input_sequence_1 = np.array([[1.0], [2.0], [3.0]])
    initial_hidden_state_1 = np.zeros((1, 1))
    initial_cell_state_1 = np.zeros((1, 1))
    
    lstm_1 = LSTM(input_size=1, hidden_size=1)
    # Set weights and biases for reproducibility
    lstm_1.Wf = np.array([[0.5, 0.5]])
    lstm_1.Wi = np.array([[0.5, 0.5]])
    lstm_1.Wc = np.array([[0.3, 0.3]])
    lstm_1.Wo = np.array([[0.5, 0.5]])
    lstm_1.bf = np.array([[0.1]])
    lstm_1.bi = np.array([[0.1]])
    lstm_1.bc = np.array([[0.1]])
    lstm_1.bo = np.array([[0.1]])
    
    outputs_1, final_h_1, final_c_1 = lstm_1.forward(input_sequence_1, initial_hidden_state_1, initial_cell_state_1)
    
    assert np.allclose(final_h_1, np.array([[0.73698596]]), atol=1e-4), f"Test case 1 failed: expected [[0.7216]], got {final_h_1}"
    
    # Test case 2: Multiple feature inputs
    input_sequence_2 = np.array([[0.1, 0.2], [0.3, 0.4]])
    initial_hidden_state_2 = np.zeros((2, 1))
    initial_cell_state_2 = np.zeros((2, 1))
    
    lstm_2 = LSTM(input_size=2, hidden_size=2)
    # Set weights and biases for reproducibility
    lstm_2.Wf = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    lstm_2.Wi = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    lstm_2.Wc = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    lstm_2.Wo = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    lstm_2.bf = np.array([[0.1], [0.2]])
    lstm_2.bi = np.array([[0.1], [0.2]])
    lstm_2.bc = np.array([[0.1], [0.2]])
    lstm_2.bo = np.array([[0.1], [0.2]])
    
    outputs_2, final_h_2, final_c_2 = lstm_2.forward(input_sequence_2, initial_hidden_state_2, initial_cell_state_2)
    
    expected_final_h_2 = np.array([[0.16613133], [0.40299449]])
    assert np.allclose(final_h_2, expected_final_h_2, atol=1e-4), f"Test case 2 failed: expected {expected_final_h_2}, got {final_h_2}"

if __name__ == "__main__":
    test_lstm()
    print("All LSTM tests passed.")