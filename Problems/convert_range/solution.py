import numpy as np


def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    """
    Shifts an arbitrary range [values.min(), values.max()] to a desired [c, d]

    Parameters
    ----------
    values : np.ndarray
        An array of random values we want to shift
    c: float
        Supremum of the desired range
    d: float
        Infimum of the desired range

    Returns
    -------
    values_shifted : np.ndarray
        Original values shifted to [c, d] range
    """

    a, b = values.min(), values.max()
    return c + (d-c) / (b-a) * (values-a)


def test_convert_range():
    # Test 1
    seq = np.array([388, 242, 124, 384, 313, 277, 339, 302, 268, 392])
    c, d = 0, 1
    e_shifted = np.array([
        0.98507463, 0.44029851, 0., 0.97014925, 0.70522388, 0.57089552,
        0.80223881, 0.6641791, 0.53731343, 1.])
    assert np.allclose(e_shifted,convert_range(seq, c, d)), 'Test case 1 failed'

    # Test 2
    seq = np.array([[2028, 4522], [1412, 2502], [3414, 3694], [1747, 1233], [1862, 4868]])
    c, d = 4, 8
    e_shifted = np.array(
        [[4.874828060522696, 7.619257221458047], [4.19697386519945, 5.396423658872077], 
         [6.4, 6.7081155433287485], [4.565612104539202, 4.0], [4.692159559834938, 8.0]])
    assert np.allclose(e_shifted,convert_range(seq, c, d)), 'Test case 2 failed'

    print('All tests passed')


if __name__ == '__main__':
    test_convert_range()

