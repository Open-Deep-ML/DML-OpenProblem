## Problem

Write a Python function `multi_token_prediction(hidden_states, target_ids, embedding_weight, proj_weights, lm_head_weight, num_future)` that implements the Multi-Token Prediction (MTP) training objective. The function takes:
- `hidden_states`: array of shape `(seq_len, hidden_dim)` — the main model's last hidden states
- `target_ids`: array of shape `(seq_len,)` — the target token IDs (integers)
- `embedding_weight`: array of shape `(vocab_size, hidden_dim)` — token embedding matrix
- `proj_weights`: list of `num_future` weight arrays, each of shape `(hidden_dim, 2 * hidden_dim)` — projection for each MTP head
- `lm_head_weight`: array of shape `(vocab_size, hidden_dim)` — language model head weight
- `num_future`: number of future tokens to predict (int)

For each future step `k` (1 to `num_future`):
1. Get the target embedding for position shifted by `k-1`: `emb = embedding_weight[target_ids[k-1:seq_len-num_future+k-1]]`
2. Concatenate `hidden_states[:seq_len-num_future]` with `emb` along the feature dimension
3. Project through `proj_weights[k-1]` (matrix multiply)
4. Compute logits via `lm_head_weight`
5. Compute cross-entropy loss against `target_ids[k:seq_len-num_future+k]`

Return the average loss across all steps as a float. Only use numpy.
