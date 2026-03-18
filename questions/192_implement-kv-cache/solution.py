import numpy as np

class KVCache:
    def __init__(self):
        self.keys = None
        self.values = None

    def update(self, new_k, new_v):
        if self.keys is None:
            self.keys = new_k.copy()
            self.values = new_v.copy()
        else:
            self.keys = np.concatenate([self.keys, new_k], axis=0)
            self.values = np.concatenate([self.values, new_v], axis=0)
        return self.keys, self.values

    def get(self):
        if self.keys is None:
            return None, None
        return self.keys, self.values

def attention_with_kv_cache(q, k, v, cache):
    full_k, full_v = cache.update(k, v)
    head_dim = q.shape[-1]
    scores = np.matmul(q, full_k.T) / np.sqrt(head_dim)
    attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
    output = np.matmul(attn_weights, full_v)
    return output.astype(float), cache
