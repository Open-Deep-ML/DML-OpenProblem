## Task: Compute Direct Preference Optimization Loss

Implement a function to compute the Direct Preference Optimization (DPO) loss, a technique used to fine-tune language models based on human preferences without requiring a separate reward model.

Given policy log probabilities for chosen and rejected responses, along with reference model log probabilities, your function should compute the DPO loss using the Bradley-Terry preference model.

The function should take the following inputs:
- `policy_chosen_logps`: Log probabilities from the policy model for chosen responses
- `policy_rejected_logps`: Log probabilities from the policy model for rejected responses
- `reference_chosen_logps`: Log probabilities from the reference model for chosen responses
- `reference_rejected_logps`: Log probabilities from the reference model for rejected responses
- `beta`: Temperature parameter controlling the strength of the KL constraint (default: 0.1)

Return the average DPO loss across all examples as a float.
