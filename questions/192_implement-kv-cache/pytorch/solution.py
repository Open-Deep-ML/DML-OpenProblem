import torch
import torch.nn.functional as F

class KVCache:
    def __init__(self):
        self.keys = None
        self.values = None

    def update(self, new_k, new_v):
        if self.keys is None:
            self.keys = new_k.clone()
            self.values = new_v.clone()
        else:
            self.keys = torch.cat([self.keys, new_k], dim=0)
            self.values = torch.cat([self.values, new_v], dim=0)
        return self.keys, self.values

    def get(self):
        if self.keys is None:
            return None, None
        return self.keys, self.values

def attention_with_kv_cache(q, k, v, cache):
    full_k, full_v = cache.update(k, v)
    head_dim = q.shape[-1]
    scores = torch.matmul(q, full_k.T) / (head_dim ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, full_v)
    return output.float(), cache
