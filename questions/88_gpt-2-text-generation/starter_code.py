def gen_text(prompt: str, n_tokens_to_generate: int = 40):
    ####your code goes here####
    pass


def load_encoder_hparams_and_params(
    model_size: str = "124M", models_dir: str = "models"
):
    class DummyBPE:
        def __init__(self):
            self.encoder_dict = {"hello": 1, "world": 2, "<UNK>": 0}

        def encode(self, text: str):
            tokens = text.strip().split()
            return [
                self.encoder_dict.get(token, self.encoder_dict["<UNK>"])
                for token in tokens
            ]

        def decode(self, token_ids: list):
            reversed_dict = {v: k for k, v in self.encoder_dict.items()}
            return " ".join(
                [reversed_dict.get(tok_id, "<UNK>") for tok_id in token_ids]
            )

    hparams = {"n_ctx": 1024, "n_head": 12}

    params = {
        "wte": np.random.rand(3, 10),
        "wpe": np.random.rand(1024, 10),
        "blocks": [],
        "ln_f": {
            "g": np.ones(10),
            "b": np.zeros(10),
        },
    }

    encoder = DummyBPE()
    return encoder, hparams, params
