def gini_impurity(y: list[int]) -> float:
    classes = set(y)
    n = len(y)

    gini_impurity = 0

    for cls in classes:
        gini_impurity += (y.count(cls) / n) ** 2

    return round(1 - gini_impurity, 3)
