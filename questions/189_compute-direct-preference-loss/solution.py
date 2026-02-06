import numpy as np

def compute_dpo_loss(policy_chosen_logps: np.ndarray, policy_rejected_logps: np.ndarray,
                     reference_chosen_logps: np.ndarray, reference_rejected_logps: np.ndarray,
                     beta: float = 0.1) -> float:
    """
    Compute the Direct Preference Optimization (DPO) loss.

    Args:
        policy_chosen_logps: Log probabilities from policy model for chosen responses
        policy_rejected_logps: Log probabilities from policy model for rejected responses
        reference_chosen_logps: Log probabilities from reference model for chosen responses
        reference_rejected_logps: Log probabilities from reference model for rejected responses
        beta: Temperature parameter (default: 0.1)

    Returns:
        Average DPO loss across all examples
    """
    # Compute log ratios for chosen and rejected responses
    policy_log_ratios = policy_chosen_logps - policy_rejected_logps
    reference_log_ratios = reference_chosen_logps - reference_rejected_logps

    # Compute the logits for the Bradley-Terry model
    logits = beta * (policy_log_ratios - reference_log_ratios)

    # Compute loss using log-sigmoid for numerical stability
    # Loss = -log(sigmoid(logits)) = log(1 + exp(-logits))
    losses = np.log1p(np.exp(-logits))

    # Return the average loss
    return float(np.mean(losses))
