import numpy as np

def multi_token_prediction(hidden_states, target_ids, embedding_weight, proj_weights, lm_head_weight, num_future):
    seq_len, hidden_dim = hidden_states.shape
    effective_len = seq_len - num_future
    total_loss = 0.0

    for k in range(1, num_future + 1):
        emb = embedding_weight[target_ids[k-1:effective_len+k-1]]
        concat = np.concatenate([hidden_states[:effective_len], emb], axis=-1)
        projected = concat @ proj_weights[k-1].T
        logits = projected @ lm_head_weight.T

        targets = target_ids[k:effective_len+k]
        logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        log_sum_exp = np.log(np.sum(exp_logits, axis=-1))
        correct_logits = logits_shifted[np.arange(effective_len), targets]
        loss = -np.mean(correct_logits - log_sum_exp)
        total_loss += loss

    return float(total_loss / num_future)
