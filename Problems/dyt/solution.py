import numpy as np


def dynamic_tanh(x: np.ndarray, alpha: float, gamma: float, beta: float) -> list[float]:
    """
    Applies DyT to an array. Could serve as a replacement
    for layer normalization in Transformers. 

    Parameters
    ----------
    x : np.ndarray
        Input tensor of shape (B,T,C)
    alpha : float
        Learnable parameter of the DyT layer
    gamma : float 
        Learnable scaling parameter vector of shape (C, ) of the DyT layer
    beta : float
        Learnable scaling parameter vector of shape (C, ) of the DyT layer
    eps : float
        Epsilon constant

    Returns
    -------
    x : list[float]
        Input x with DyT applied to it and rounded up to 4 floating points
    """

    def tanh(x: np.ndarray) -> np.ndarray:
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    x = tanh(alpha * x)
    return (x * gamma + beta).round(4).tolist()


def test_dynamic_tanh():
    alpha = .5

    # Test 1
    x = np.array([[[0.14115588, 0.00372817, 0.24126647, 0.22183601],
        [0.36301332, 0.67681456, 0.3723281 , 0.62767559],
        [0.94926205, 0.80230257, 0.19737574, 0.04460771],
        [0.43777021, 0.95744001, 0.60795979, 0.58980314],
        [0.27250625, 0.48053656, 0.11087151, 0.06228769]],
       [[0.12620219, 0.63002473, 0.75673539, 0.60411435],
        [0.3918192 , 0.39810709, 0.42186426, 0.79954607],
        [0.67730682, 0.96539769, 0.13366266, 0.44462357],
        [0.31556188, 0.86050486, 0.96060468, 0.43953706],
        [0.80002165, 0.39582123, 0.35731605, 0.83600622]]])
    gamma, beta = np.ones(shape=(x.shape[2])), np.zeros(shape=(x.shape[2]))
    expected_x = [[[0.0705, 0.0019, 0.1201, 0.1105],
        [0.1795, 0.3261, 0.184, 0.3039],
        [0.4419, 0.3809, 0.0984, 0.0223],
        [0.2155, 0.4452, 0.295, 0.2866],
        [0.1354, 0.2357, 0.0554, 0.0311]],
        [[0.063, 0.305, 0.3613, 0.2932],
        [0.1934, 0.1965, 0.2079, 0.3798],
        [0.3263, 0.4484, 0.0667, 0.2187],
        [0.1565, 0.4055, 0.4465, 0.2163],
        [0.38, 0.1954, 0.1768, 0.3952]]]
    output_x = dynamic_tanh(x, alpha, gamma, beta)
    assert expected_x == output_x, 'Test case 1 failed'

    # Test 2
    x = np.array([[[0.20793482, 0.16989285, 0.03898972],
        [0.17912554, 0.10962205, 0.3870742],
        [0.00107181, 0.35807922, 0.15861333]]])
    gamma, beta = np.ones(shape=(x.shape[2])), np.zeros(shape=(x.shape[2]))
    expected_x = [[[0.1036, 0.0847, 0.0195],
        [0.0893, 0.0548, 0.1912],
        [0.0005, 0.1772, 0.0791]]]
    output_x = dynamic_tanh(x, alpha, gamma, beta)
    assert expected_x == output_x, 'Test case 2 failed'

    # Test 3
    x = np.array([[[0.94378259]],[[0.97754654]],[[0.36168351]],[[0.51821078]],[[0.76961589]]])
    gamma, beta = np.ones(shape=(x.shape[2])), np.zeros(shape=(x.shape[2]))
    expected_x = [[[0.4397]],[[0.4532]],[[0.1789]],[[0.2535]],[[0.3669]]]
    output_x = dynamic_tanh(x, alpha, gamma, beta)
    assert expected_x == output_x, 'Test case 3 failed'

    print('All tests passed')


if __name__ == '__main__':
    test_dynamic_tanh()