class GroupQueryAttention:
    def __init__(self, head_dim, num_query_heads, num_key_value_heads, output_dim = None):

        if num_query_heads % num_key_value_heads!= 0:
            raise ValueError(f"Number of query heads must be divisible by number of key value heads. ",
                             f"Found `num_query_heads` = {num_query_heads}, `num_key_value_heads` = {num_key_value_heads}")

        self._head_dim = head_dim
        self._num_query_heads = num_query_heads
        self._num_key_value_heads = num_key_value_heads
        self._output_dim = output_dim

        self._sqrt_key_dim = np.sqrt(head_dim)
        self._num_repeats = num_query_heads // num_key_value_heads

        self.built = False

    def build(self, input_shapes):
        q_shape, k_shape, v_shape = input_shapes

        assert v_shape == k_shape, f"Shape of key, value must be same. Found key shape: {k_shape}, value shape: {v_shape}"

        self.q_proj = Dense(self._num_query_heads * self._head_dim, kernel_init_scale = 0.1, use_bias = False)
        self.k_proj = Dense(self._num_key_value_heads * self._head_dim, kernel_init_scale = 0.1, use_bias = False)
        self.v_proj = Dense(self._num_key_value_heads * self._head_dim, kernel_init_scale = 0.1, use_bias = False)
        if self._output_dim is not None:
            self._output_proj = Dense(self._output_dim, kernel_init_scale = 0.1, use_bias = False)
        else:
            self._output_proj = Dense(q_shape[-1], kernel_init_scale = 0.1, use_bias = False)

        self.built = True


    def _compute_attention(self, query, value, key, attention_mask = None):
        # query shape: (B, T, ft_dim)
        # value shape: (B, S, ft_dim)
        # key shape:   (B, S, ft_dim)
        # head dim: dim
        # number of query heads: Nq, number of key value heads = Nk

        B, T, _ = query.shape
        _, S1, _ = key.shape; _, S2, _ = value.shape

        assert S1 == S2, f"Shapes of key and value must match along dimension 1. Found key.shape[1] = {S1} and value.shape[1] = {S2}"

        S = S1
        del S1, S2

        query = self.q_proj(query).reshape((B, T, self._num_query_heads, self._head_dim)).transpose(0, 2, 1, 3)       # Shape: (B, Nq, T, dim)
        key = self.k_proj(key).reshape((B, S, self._num_key_value_heads, self._head_dim)).transpose(0, 2, 1, 3)       # Shape: (B, Nk, S, dim)
        value = self.v_proj(value).reshape((B, S, self._num_key_value_heads, self._head_dim)).transpose(0, 2, 1, 3)   # Shape: (B, Nk, S, dim)

        key = np.repeat(key, self._num_repeats, axis = 1)           # Shape: (B, Nq, S, dim)
        value = np.repeat(value, self._num_repeats, axis = 1)       # Shape: (B, Nq, S, dim)

        # ---- Variables ---- 
        # b: batch_size
        # n: Nq
        # q: T
        # k: S
        # d: head_dim (= dim)

        attn_score_eqn = 'bnqd, bnkd -> bnqk'
        attn_op_eqn = 'bnqk, bnkd -> bnqd'

        query /= self._sqrt_key_dim

        attn_scores = np.einsum(attn_score_eqn, query, key)         # Shape: (B, Nq, T, S)
        attn_scores = softmax(attn_scores, axis = -1)

        if attention_mask is not None:
            attn_scores *= attention_mask

        attention_op = np.einsum(attn_op_eqn, attn_scores, value)                       # Shape: (B, Nq, T, dim)
        attention_op = attention_op.transpose(0, 2, 1, 3).reshape((B, T, -1))           # Shape: (B, T, Nq * dim)
        attention_op = self._output_proj(attention_op)                                  # Shape: (B, Nq, T, output_dim)

        return attn_scores, attention_op

    def __call__(self, query, value, key = None, attention_mask = None, return_attention_scores = False):
        key = value if key is None else key
        if not self.built:
            self.build([query.shape, key.shape, value.shape])

        attention_scores, attention_op = self._compute_attention(query, value, key, attention_mask = attention_mask)

        if not return_attention_scores:
            return attention_op

        return attention_scores, attention_op