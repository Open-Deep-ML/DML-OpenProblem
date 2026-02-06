import numpy as np

def compute_dpo_loss(policy_chosen_logps: np.ndarray, policy_rejected_logps: np.ndarray,
                     reference_chosen_logps: np.ndarray, reference_rejected_logps: np.ndarray,
                     beta: float = 0.1) -> float:
    # Your code here
    pass
