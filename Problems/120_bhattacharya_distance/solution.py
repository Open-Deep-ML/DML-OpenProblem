import numpy as np

def bhattacharyya_distance(p : list[float], q : list[float]) -> float:

    if len(p) != len(q) :
        return 0.0
    
    p, q = np.array(p), np.array(q)

    BC = np.sum(np.sqrt(p * q))    #### Bhattacharya coefficient

    DB = -np.log(BC)               #### Bhattacharya distance

    return round(DB, 4)

def test_bhattacharyya_distance() -> None:

    # Test Case 1
    p = [0.1, 0.2, 0.3, 0.4]
    q = [0.4, 0.3, 0.2, 0.1]
    assert bhattacharyya_distance(p, q) == 0.1166

    # Test Case 2
    p = [0.7, 0.2, 0.1]
    q = [0.4, 0.3, 0.3]
    assert bhattacharyya_distance(p, q) == 0.0541

    # Test Case 3
    p = []
    q = [0.5, 0.4, 0.1]
    assert bhattacharyya_distance(p, q) == 0.0

    # Test Case 4
    p = [0.6, 0.4]
    q = [0.1, 0.7, 0.2]
    assert bhattacharyya_distance(p, q) == 0.0

    # Test Case 5
    p = [0.6, 0.2, 0.1, 0.1]
    q = [0.1, 0.2, 0.3, 0.4]
    assert bhattacharyya_distance(p, q) == 0.2007

if __name__ == '__main__':

    test_bhattacharyya_distance()
    print('All Bhattacharyya Distance test cases passed')



