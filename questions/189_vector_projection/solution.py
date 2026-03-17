def project_vector(a, b):
    """
    Compute the projection of vector a onto vector b.

    Args:
        a (list): A list of numbers representing the first vector.
        b (list): A list of numbers representing the vector onto which a is projected.

    Returns:
        list: The projection of a onto b.

    Raises:
        ValueError: If the vectors do not have the same length.
        ValueError: If b is the zero vector.
    """
    dot_ab = sum(x * y for x, y in zip(a, b))
    dot_bb = sum(y * y for y in b)

    if dot_bb == 0:
        raise ValueError("Cannot project onto the zero vector")

    scale = dot_ab / dot_bb
    return [scale * y for y in b]