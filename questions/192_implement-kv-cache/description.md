## Problem

Write a Python class `KVCache` that implements a key-value cache for efficient autoregressive inference in transformers. The class should support:
- `__init__(self)`: Initialize an empty cache.
- `update(self, new_k, new_v)`: Append new key and value arrays along the sequence dimension (axis 0) and return the full cached `(keys, values)`.
- `get(self)`: Return the current cached `(keys, values)` tuple, or `(None, None)` if empty.

Then write a function `attention_with_kv_cache(q, k, v, cache)` that computes scaled dot-product attention using the cache. It should update the cache with the new `k`, `v`, compute attention using the full cached keys/values, and return `(output, cache)`. Arrays have shape `(seq_len, head_dim)`. Only use numpy.
