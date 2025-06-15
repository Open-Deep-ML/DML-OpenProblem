def OSA(source: str, target: str) -> int:
    source_len, target_len = len(source), len(target)
    
    osa_matrix = [[0]*(target_len + 1) for _ in range(source_len + 1)]
    
    for j in range(1, target_len + 1):
        osa_matrix[0][j] = j
    
    for i in range(1, source_len + 1):
        osa_matrix[i][0] = i
    
    for i in range(1, source_len + 1):
        for j in range(1, target_len + 1):
            osa_matrix[i][j] = min(
                osa_matrix[i - 1][j] + 1,
                osa_matrix[i][j - 1] + 1,
                osa_matrix[i - 1][j - 1] + (1 if source[i - 1] != target[j - 1] else 0)
            )
            if i > 1 and j > 1 and source[i - 1] == target[j-2] and source[i-2] == target[j - 1]:
                osa_matrix[i][j] = min(osa_matrix[i][j], osa_matrix[i - 2][j - 2] + 1)

    return osa_matrix[-1][-1]

def test_OSA():

    input_string_pairs = [
    ("butterfly", "dragonfly"),
    ("london", "paris"),
    ("shower", "grower"),
    ("telescope", "microscope"),
    ("labyrinth", "puzzle"),
    ("silhouette", "shadow"),
    ("whisper", "screaming"),
    ("enigma", "mystery"),
    ("symphony", "cacophony"),
    ("mirage", "oasis"),
    ("asteroid", "meteorite"),
    ("palindrome", "palladium"),
    ("caper", "acer")
    ]

    expected_output = [6, 6, 2, 5, 9, 8, 9, 7, 4, 6, 5, 5, 2]

    for (s1, s2), expected_distance in zip(input_string_pairs, expected_output):
        assert OSA(s1, s2) == expected_distance

if __name__ == "__main__":
    test_OSA()
    print("All OSA distance tests passed")
