import numpy as np

def compute_pmi(joint_counts, total_counts_x, total_counts_y, total_samples):

    if not all(isinstance(x, (int, float)) for x in [joint_counts, total_counts_x, total_counts_y, total_samples]):
        raise ValueError("All inputs must be numeric")

    if any(x < 0 for x in [joint_counts, total_counts_x, total_counts_y, total_samples]):
        raise ValueError("Counts cannot be negative")

    if total_samples == 0:
        raise ValueError("Total samples cannot be zero")

    if joint_counts > min(total_counts_x, total_counts_y):
        raise ValueError("Joint counts cannot exceed individual counts")

    if any(x > total_samples for x in [total_counts_x, total_counts_y]):
        raise ValueError("Individual counts cannot exceed total samples")

    p_x = total_counts_x / total_samples
    p_y = total_counts_y / total_samples
    p_xy = joint_counts / total_samples

    # Handle edge cases where probabilities are zero
    if p_xy == 0 or p_x == 0 or p_y == 0:
        return float('-inf')

    pmi = np.log2(p_xy / (p_x * p_y))

    return round(pmi, 3)

def test_pmi():
    # Test Case 1: Perfect positive association
    joint_counts1 = 100
    total_counts_x1 = 100
    total_counts_y1 = 100
    total_samples1 = 100
    expected1 = round(np.log2(1/(1*1)), 3)  # Should be 0.0
    assert compute_pmi(joint_counts1, total_counts_x1, total_counts_y1, total_samples1) == expected1, "Test Case 1 Failed"

    # Test Case 2: Independence
    joint_counts2 = 25
    total_counts_x2 = 50
    total_counts_y2 = 50
    total_samples2 = 100
    expected2 = round(np.log2((25/100)/((50/100)*(50/100))), 3)  # Should be 0.0
    assert compute_pmi(joint_counts2, total_counts_x2, total_counts_y2, total_samples2) == expected2, "Test Case 2 Failed"

    # Test Case 3: Negative association
    joint_counts3 = 10
    total_counts_x3 = 50
    total_counts_y3 = 50
    total_samples3 = 100
    expected3 = round(np.log2((10/100)/((50/100)*(50/100))), 3)  # Should be negative
    assert compute_pmi(joint_counts3, total_counts_x3, total_counts_y3, total_samples3) == expected3, "Test Case 3 Failed"

    # Test Case 4: Zero joint occurrence
    joint_counts4 = 0
    total_counts_x4 = 50
    total_counts_y4 = 50
    total_samples4 = 100
    expected4 = float('-inf')
    assert compute_pmi(joint_counts4, total_counts_x4, total_counts_y4, total_samples4) == expected4, "Test Case 4 Failed"

    # Test Case 5: Invalid inputs
    try:
        compute_pmi(-1, 50, 50, 100)
        assert False, "Test Case 5 Failed: Should raise ValueError for negative counts"
    except ValueError:
        pass

    print("All Test Cases Passed!")

if __name__ == "__main__":
    test_pmi()
