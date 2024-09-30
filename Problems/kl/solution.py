import numpy as np

def kl_divergence_normal(mu_p: float, sigma_p: float, mu_q: float, sigma_q: float) -> float:
    """
    Computes the KL divergence between two normal distributions P and Q.
    P ~ N(mu_p, sigma_p^2)
    Q ~ N(mu_q, sigma_q^2)
    """
    term1 = np.log(sigma_q / sigma_p)
    term2 = (sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2)
    kl_div = term1 + term2 - 0.5
    return kl_div

def test_kl_divergence_normal():
    # Test case 1: Identical distributions
    mu_p1, sigma_p1 = 0.0, 1.0
    mu_q1, sigma_q1 = 0.0, 1.0
    expected_kl1 = 0.0
    output_1 = kl_divergence_normal(mu_p1, sigma_p1, mu_q1, sigma_q1)
    assert np.allclose(output_1, expected_kl1, atol=1e-6), f"Test case 1 failed: expected {expected_kl1}, got {output_1}"

    # Test case 2: Different means
    mu_p2, sigma_p2 = 0.0, 1.0
    mu_q2, sigma_q2 = 1.0, 1.0
    expected_kl2 = 0.5  # Calculated as ((mu_p - mu_q)^2) / (2 * sigma_q^2)
    output_2 = kl_divergence_normal(mu_p2, sigma_p2, mu_q2, sigma_q2)
    assert np.allclose(output_2, expected_kl2, atol=1e-6), f"Test case 2 failed: expected {expected_kl2}, got {output_2}"

    # Test case 3: Different variances
    mu_p3, sigma_p3 = 0.0, 1.0
    mu_q3, sigma_q3 = 0.0, 2.0
    expected_kl3 = np.log(2 / 1) + (1**2 + (0 - 0)**2) / (2 * 2**2) - 0.5
    output_3 = kl_divergence_normal(mu_p3, sigma_p3, mu_q3, sigma_q3)
    assert np.allclose(output_3, expected_kl3, atol=1e-6), f"Test case 3 failed: expected {expected_kl3}, got {output_3}"

    # Test case 4: Different means and variances
    mu_p4, sigma_p4 = 1.0, 1.0
    mu_q4, sigma_q4 = 0.0, 2.0
    expected_kl4 = np.log(2 / 1) + (1**2 + (1 - 0)**2) / (2 * 2**2) - 0.5
    output_4 = kl_divergence_normal(mu_p4, sigma_p4, mu_q4, sigma_q4)
    assert np.allclose(output_4, expected_kl4, atol=1e-6), f"Test case 4 failed: expected {expected_kl4}, got {output_4}"

if __name__ == "__main__":
    test_kl_divergence_normal()
    print("All KL divergence tests passed.")
