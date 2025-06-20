import numpy as np


def grpo_objective(
    rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01
) -> float:
    """
    Compute the GRPO objective function.

    Args:
        rhos: List of likelihood ratios (ρ_i) = π_theta(o_i | q) / π_theta_old(o_i | q).
        A: List of advantage estimates (A_i).
        pi_theta_old: List representing the old policy probabilities π_theta_old(o_i | q).
        pi_theta_ref: List representing the reference policy probabilities π_ref(o_i | q).
        epsilon: Clipping parameter (ϵ).
        beta: KL divergence penalty coefficient (β).

    Returns:
        The computed GRPO objective value.
    """
    G = len(rhos)
    if not (len(A) == len(pi_theta_old) == len(pi_theta_ref) == G):
        raise ValueError("All input lists must have the same length.")

    # Compute clipped likelihood ratios
    clipped_rhos = np.clip(rhos, 1 - epsilon, 1 + epsilon)

    # Compute the minimum terms for the objective
    unclipped = np.array(rhos) * np.array(A)
    clipped = clipped_rhos * np.array(A)
    min_terms = np.minimum(unclipped, clipped)
    average_min = np.mean(min_terms)

    # Compute pi_theta from rhos and pi_theta_old
    pi_theta = np.array(rhos) * np.array(pi_theta_old)

    # Normalize pi_theta and pi_theta_ref to ensure they are valid probability distributions
    pi_theta /= np.sum(pi_theta)
    pi_theta_ref /= np.sum(pi_theta_ref)

    # Compute KL divergence D_KL(pi_theta || pi_theta_ref)
    kl_divergence = np.sum(
        pi_theta * np.log(pi_theta / pi_theta_ref + 1e-10)
    )  # Added epsilon to avoid log(0)

    # Compute the final objective
    objective = average_min - beta * kl_divergence

    return objective
