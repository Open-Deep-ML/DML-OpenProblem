import numpy as np

def pos_encoding(position:int, dmodel:int):
    pos = np.array(np.arange(position), np.float32)
    ind = np.array(np.arange(dmodel), np.float32)
    pos = pos.reshape(position,1)
    ind = ind.reshape(1,dmodel)


    def get_angles(pos, i):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dmodel))
        return pos * angles

    angle1 = get_angles(pos, ind)

    sine = np.sin(angle1[:, 0::2])
    cosine = np.cos(angle1[:, 1::2])

    pos_encoding = np.concatenate([sine, cosine], axis = -1)
    pos_encoding.shape
    pos_encoding = pos_encoding[np.newaxis, :]
    pos_encoding.shape
    pos_encoding = np.float32(pos_encoding)
    
    return pos_encoding

def test_pos_encoding() -> None:
    # Test case 1:
    ans1 = (1, 2, 8)
    assert pos_encoding(2,8).shape == ans1, "Test case 1 failed"
    
    # Test case 2:
    ans2 = (1, 10, 512)
    assert pos_encoding(10, 512).shape == ans2, "Test case 2 fialed"
    
    # Test case 3:
    ans3 = (1, 0, 0)
    assert pos_encoding(0, 0).shape == ans3, "Test case 3 failed"

if __name__ == "__main__":
    test_pos_encoding()
    print("All tests passed.")
