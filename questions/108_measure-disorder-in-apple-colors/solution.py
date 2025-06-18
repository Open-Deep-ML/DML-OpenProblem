def disorder(apples: list) -> float:
    """
    Calculates a measure of disorder in a basket of apples based on their colors.
    One valid approach is to use the Gini impurity, defined as:
      G = 1 - sum((count/total)^2 for each color)
    This method returns 0 for a basket with all apples of the same color and increases as the variety of colors increases.
    While this implementation uses the Gini impurity, any method that satisfies the following properties is acceptable:
      1. A single color results in a disorder of 0.
      2. Baskets with more distinct colors yield a higher disorder score.
      3. The ordering constraints are maintained.
    """
    if not apples:
        return 0.0
    total = len(apples)
    counts = {}
    for color in apples:
        counts[color] = counts.get(color, 0) + 1
    impurity = 1.0
    for count in counts.values():
        p = count / total
        impurity -= p * p
    return round(impurity, 4)
