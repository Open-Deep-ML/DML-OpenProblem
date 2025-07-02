def grpo_objective(
    rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01
) -> float:
    """
    Compute the GRPO objective function.

    Args:
            rhos: List of likelihood ratios (p_i) = pi_theta(o_i | q) / pi_theta_old(o_i | q).
            A: List of advantage estimates (A_i).
            pi_theta_old: List representing the old policy probabilities pi_theta_old(o_i | q).
            pi_theta_ref: List representing the reference policy probabilities pi_ref(o_i | q).
            epsilon: Clipping parameter (eps).
            beta: KL divergence penalty coefficient (beta).

    Returns:
            The computed GRPO objective value.
    """
    # Your code here
    pass
