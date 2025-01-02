import numpy as np
np.random.seed(42)
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, g, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b

def linear(x, w, b):
    return x @ w + b

def ffn(x, c_fc, c_proj):
    return linear(gelu(linear(x, **c_fc)), **c_proj)

def attention(q, k, v, mask):
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v

def mha(x, c_attn, c_proj, n_head):
    x = linear(x, **c_attn)
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), np.split(x, 3, axis=-1)))
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    x = linear(np.hstack(out_heads), **c_proj)
    return x

def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x

def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    x = wte[inputs] + wpe[range(len(inputs))]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    return layer_norm(x, **ln_f) @ wte.T

def generate(inputs, params, n_head, n_tokens_to_generate):
    for _ in range(n_tokens_to_generate):
        logits = gpt2(inputs, **params, n_head=n_head)
        next_id = np.argmax(logits[-1])
        inputs.append(int(next_id))
    return inputs[len(inputs) - n_tokens_to_generate:]

def gen_text(prompt: str, n_tokens_to_generate: int = 40):
    np.random.seed(42)  # Set the random seed for reproducibility
    encoder, hparams, params = load_encoder_hparams_and_params()
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    output_text = encoder.decode(output_ids)
    return output_text

def load_encoder_hparams_and_params(model_size: str = "124M", models_dir: str = "models"):
    class DummyBPE:
        def __init__(self):
            self.encoder_dict = {"hello": 1, "world": 2, "<UNK>": 0}

        def encode(self, text: str):
            tokens = text.strip().split()
            return [self.encoder_dict.get(token, self.encoder_dict["<UNK>"]) for token in tokens]

        def decode(self, token_ids: list):
            reversed_dict = {v: k for k, v in self.encoder_dict.items()}
            return " ".join([reversed_dict.get(tok_id, "<UNK>") for tok_id in token_ids])

    hparams = {
        "n_ctx": 1024,
        "n_head": 12
    }

    params = {
        "wte": np.random.rand(3, 10),
        "wpe": np.random.rand(1024, 10),
        "blocks": [],
        "ln_f": {
            "g": np.ones(10),
            "b": np.zeros(10),
        }
    }

    encoder = DummyBPE()
    return encoder, hparams, params

def test_gen_text() -> None:
    # Test case 1
    result1 = gen_text("hello", n_tokens_to_generate=5)
    expected1 = "hello hello hello <UNK> <UNK>"
    assert result1 == expected1, f"Test case 1 failed: {result1} != {expected1}"

    # Test case 2
    result2 = gen_text("hello world", n_tokens_to_generate=10)
    expected2 = "world world world world world world world world world world"
    assert result2 == expected2, f"Test case 2 failed: {result2} != {expected2}"

    # Test case 3
    result3 = gen_text("world", n_tokens_to_generate=3)
    expected3 = "world world world"
    assert result3 == expected3, f"Test case 3 failed: {result3} != {expected3}"

if __name__ == "__main__":
    test_gen_text()
    print("All tests passed.")
