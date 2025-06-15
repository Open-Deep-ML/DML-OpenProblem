import numpy as np

def pos_encoding(position:int, dmodel:int):
    
    if position == 0 or dmodel == 0: return -1

    pos = np.array(np.arange(position), np.float16)
    ind = np.array(np.arange(dmodel), np.float16)
    pos = pos.reshape(position,1)
    ind = ind.reshape(1,dmodel)


    def get_angles(pos, i):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float16(dmodel))
        return pos * angles

    angle1 = get_angles(pos, ind)

    sine = np.sin(angle1[:, 0::2])
    cosine = np.cos(angle1[:, 1::2])

    pos_encoding = np.concatenate([sine, cosine], axis = -1)
    pos_encoding = pos_encoding[np.newaxis, :]
    pos_encoding = np.float16(pos_encoding)

    return pos_encoding

def test_pos_encoding() -> None:
    # Test case 1:
    ans1 = np.array([[[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                      [0.8413, 0.0998, 0.01, 0.001, 0.5405, 0.995, 1.0, 1.0]]], dtype=np.float16)
    result1 = pos_encoding(2, 8)
    assert np.allclose(result1, ans1, atol=1e-3), f"Test case 1 failed: {result1} != {ans1}"


    # Test case 2:
    ans2 = np.array(
        [[[ 0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
    0.000e+00,  0.000e+00,  1.000e+00,  1.000e+00,  1.000e+00,  1.000e+00,
    1.000e+00,  1.000e+00,  1.000e+00,  1.000e+00],
  [ 8.413e-01,  3.110e-01,  9.985e-02,  3.162e-02,  1.000e-02,  3.162e-03,
    1.000e-03,  3.161e-04,  5.405e-01,  9.502e-01,  9.951e-01,  9.995e-01,
    1.000e+00,  1.000e+00,  1.000e+00,  1.000e+00],
  [ 9.092e-01,  5.913e-01,  1.986e-01,  6.323e-02,  2.000e-02,  6.325e-03,
    2.001e-03,  6.323e-04, -4.163e-01,  8.066e-01,  9.800e-01,  9.980e-01,
    1.000e+00,  1.000e+00,  1.000e+00,  1.000e+00],
  [ 1.411e-01,  8.125e-01,  2.954e-01,  9.473e-02,  3.000e-02,  9.483e-03,
    3.000e-03,  9.489e-04, -9.902e-01,  5.825e-01,  9.556e-01,  9.956e-01,
    9.995e-01,  1.000e+00,  1.000e+00,  1.000e+00],
  [-7.568e-01,  9.536e-01,  3.894e-01,  1.261e-01,  3.998e-02,  1.265e-02,
    4.002e-03,  1.265e-03, -6.538e-01,  3.010e-01,  9.209e-01,  9.922e-01,
    9.990e-01,  1.000e+00,  1.000e+00,  1.000e+00]]], 
        dtype=np.float16)
    result2 = pos_encoding(5, 16)
    assert np.allclose(result2, ans2, atol=1e-3), f"Test case 2 failed: {result2} != {ans2}"


    # Test case 3:
    ans3 = np.array([])
    assert np.allclose(pos_encoding(0,0), ans3, atol=1e-3), f"Test case 3 failed: {pos_encoding(0,0)} != {ans3}"


    # Test case 4:
    ans4 = -1
    assert np.allclose(pos_encoding(2,-1), ans4, atol=1e-3), f"Test case 4 failed: {pos_encoding(2,-1)} != {ans4}"



if __name__ == "__main__":
    test_pos_encoding()
    print("All tests passed.")