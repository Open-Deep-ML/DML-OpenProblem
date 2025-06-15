import numpy as np


def compute_pmi(joint_counts, total_counts_x, total_counts_y, total_samples):
    if not all(
        isinstance(x, int) and x >= 0
        for x in [joint_counts, total_counts_x, total_counts_y, total_samples]
    ):
        raise ValueError("All inputs must be non-negative integers.")

    if total_samples == 0:
        raise ValueError("Total samples cannot be zero.")

    if joint_counts > min(total_counts_x, total_counts_y):
        raise ValueError("Joint counts cannot exceed individual counts.")

    p_x = total_counts_x / total_samples
    p_y = total_counts_y / total_samples
    p_xy = joint_counts / total_samples

    if p_xy == 0:
        return float("-inf")

    pmi = np.log2(p_xy / (p_x * p_y))

    return round(pmi, 3)
