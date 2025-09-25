import numpy as np
from typing import List, Tuple

def predict_future_position_using_kalman_filter(
    measurements: List[Tuple[float, float]],
    t_future: float
) -> Tuple[float, float]:
    """
    measurements: list of (x, y) observations
    t_future: number of steps ahead to predict
    """
    dt = 1.0
    x0, y0 = measurements[0]
    x1, y1 = measurements[1]
    
    # Initial state: [x, vx, y, vy]
    state = np.array([[x1], [x1 - x0], [y1], [y1 - y0]])
    
    # Initial state covariance
    P = np.eye(4) * 1000.0

    # State transition matrix
    F = np.array([
        [1, dt, 0,  0],
        [0,  1, 0,  0],
        [0,  0, 1, dt],
        [0,  0, 0,  1]
    ])

    # Measurement matrix
    H = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ])

    # Process noise covariance
    Q = np.eye(4) * 1e-4

    # Measurement noise covariance
    R = np.eye(2) * 0.01

    I = np.eye(4)

    for mx, my in measurements:
        # Predict
        state = np.dot(F, state)
        P = np.dot(np.dot(F, P), F.T) + Q

        # Measurement update
        z = np.array([[mx], [my]])
        y = z - np.dot(H, state)
        S = np.dot(np.dot(H, P), H.T) + R
        K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
        state = state + np.dot(K, y)
        P = np.dot(I - np.dot(K, H), P)

    # Future prediction
    for _ in range(int(t_future)):
        state = np.dot(F, state)

    return float(state[0]), float(state[2])

def test_predict_future_position_using_kalman_filter():
    # Test 1: perfect linear motion
    m = [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert np.allclose(predict_future_position_using_kalman_filter(m, 2), (5.0, 5.0), atol=0.1)

    # Test 2: noisy diagonal motion
    m = [(0.1, 0.2), (0.9, 1.1), (2.2, 2.0), (3.1, 3.3)]
    p = predict_future_position_using_kalman_filter(m, 1)
    assert isinstance(p, tuple) and len(p) == 2

    # Test 3: stationary object
    m = [(5, 5), (5, 5), (5, 5)]
    assert np.allclose(predict_future_position_using_kalman_filter(m, 3), (5.0, 5.0), atol=0.2)

if __name__ == "__main__":
    test_predict_future_position_using_kalman_filter()
    print("All test cases passed!")
