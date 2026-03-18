## Problem

Write a Python function `grouped_query_attention(Q, K, V, num_kv_heads)` that implements Grouped Query Attention. The function takes query `Q` of shape `(num_q_heads, seq_len, head_dim)`, key `K` and value `V` of shape `(num_kv_heads, seq_len, head_dim)`, and the number of key-value heads `num_kv_heads`. The number of query heads must be divisible by `num_kv_heads`. Each KV head is shared across a group of query heads. Return the attention output of shape `(num_q_heads, seq_len, head_dim)` as floats. Only use numpy.
