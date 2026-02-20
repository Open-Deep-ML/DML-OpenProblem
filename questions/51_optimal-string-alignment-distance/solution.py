def OSA(source: str, target: str) -> int:
    source_len, target_len = len(source), len(target)

    # Initialize matrix with zeros
    osa_matrix = [[0] * (target_len + 1) for _ in range(source_len + 1)]

    # Fill the first row and first column with index values
    for j in range(1, target_len + 1):
        osa_matrix[0][j] = j
    for i in range(1, source_len + 1):
        osa_matrix[i][0] = i

    # Compute the OSA distance
    for i in range(1, source_len + 1):
        for j in range(1, target_len + 1):
            osa_matrix[i][j] = min(
                osa_matrix[i - 1][j] + 1,  # Deletion
                osa_matrix[i][j - 1] + 1,  # Insertion
                osa_matrix[i - 1][j - 1]
                + (1 if source[i - 1] != target[j - 1] else 0),  # Substitution
            )
            if (
                i > 1
                and j > 1
                and source[i - 1] == target[j - 2]
                and source[i - 2] == target[j - 1]
            ):
                osa_matrix[i][j] = min(
                    osa_matrix[i][j], osa_matrix[i - 2][j - 2] + 1
                )  # Transposition

    return osa_matrix[-1][-1]
