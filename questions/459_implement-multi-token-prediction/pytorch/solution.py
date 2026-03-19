import torch
import torch.nn.functional as F

def multi_token_prediction(hidden_states, target_ids, embedding_weight, proj_weights, lm_head_weight, num_future):
    seq_len, hidden_dim = hidden_states.shape
    effective_len = seq_len - num_future
    total_loss = 0.0

    for k in range(1, num_future + 1):
        emb = embedding_weight[target_ids[k-1:effective_len+k-1]]
        concat = torch.cat([hidden_states[:effective_len], emb], dim=-1)
        projected = concat @ proj_weights[k-1].T
        logits = projected @ lm_head_weight.T

        targets = target_ids[k:effective_len+k]
        loss = F.cross_entropy(logits, targets)
        total_loss += loss.item()

    return float(total_loss / num_future)
