import numpy as np
from collections import Counter

def meteor_score(reference, candidate, alpha=0.9, beta=3, gamma=0.5):
    if not reference or not candidate:
        raise ValueError("Reference and candidate cannot be empty")
    
    # Tokenize and count
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)
    
    # Calculate matches
    matches = sum((ref_counts & cand_counts).values())
    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)
    
    precision = matches / cand_len if cand_len > 0 else 0
    recall = matches / ref_len if ref_len > 0 else 0
    
    if matches == 0:
        return 0.0
    
    fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
    
    # Corrected chunk calculation
    chunks = 0
    i = 0
    while i < len(ref_tokens):
        if i < len(cand_tokens) and ref_tokens[i] == cand_tokens[i]:
            chunks += 1
            while i < len(ref_tokens) and i < len(cand_tokens) and ref_tokens[i] == cand_tokens[i]:
                i += 1
        else:
            i += 1
    
    # Fragmentation penalty
    penalty = gamma * ((chunks / matches) ** beta) if matches > 0 else 0
    
    # Final score
    return round(fmean * (1 - penalty), 3)

def test_meteor_score():
    # Test Case 1: Identical translations
    ref_test1 = "The cat sits on the mat"
    cand_test1 = "The cat sits on the mat"
    expected1 = 1.0
    assert meteor_score(ref_test1, cand_test1) == expected1, "Test Case Failed"
    
    # Test Case 2: Similar translations
    ref_test2 = "The quick brown fox jumps over the lazy dog"
    cand_test2 = "A quick brown fox jumps over a lazy dog"
    expected2 = 0.991
    assert meteor_score(ref_test2, cand_test2) == expected2, "Test Case Failed"
    
    # Test Case 3: Completely different translations
    ref_test3 = "The cat sits on the mat"
    cand_test3 = "Dogs run in the park"
    expected3 = 0.0
    assert meteor_score(ref_test3, cand_test3) == expected3, "Test Case Failed"
    
    # Test Case 4: Partially matching translations
    ref_test4 = "Machine learning is an exciting field"
    cand_test4 = "Machine learning algorithms are fascinating"
    expected4 = 0.667
    assert meteor_score(ref_test4, cand_test4) == expected4, "Test Case Failed"
    
    # Test Case 5: Empty input handling
    try:
        meteor_score("", "Some text")
        assert False, "Test Case Failed"
    except ValueError:
        pass
    
    # Test Case 6: Partial match with penalty
    ref_test6 = "The cat sits on the mat"
    cand_test6 = "The cat on the mat sits"
    expected6 = 0.933
    assert meteor_score(ref_test6, cand_test6) == expected6, "Test Case Failed"
    
if __name__ == "__main__":
    test_meteor_score()
    print("All Test Cases Passed!")
