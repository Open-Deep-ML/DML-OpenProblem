import numpy as np

def gini_impurity(y: list[int]) -> float:

    classes = set(y)
    n = len(y)

    gini_impurity = 0

    for cls in classes:
        gini_impurity += (y.count(cls)/n)**2

    return np.round(1-gini_impurity,3)


def test_gini_impurity() -> None:
    
    classes_1 = [0,0,0,0,1,1,1,1]
    assert gini_impurity(classes_1) == 0.5

    classes_2 = [0,0,0,0,0,1]
    assert gini_impurity(classes_2) == 0.278

    classes_3 = [0,1,2,2,2,1,2]
    assert gini_impurity(classes_3) == 0.571

if __name__ == "__main__":
    test_gini_impurity()
    print("All Gini Impurity tests passed.")
