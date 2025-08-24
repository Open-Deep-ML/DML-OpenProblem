class LoRALinear(nn.Module):
    def __init__(self, original_layer_shape: tuple, rank: int, alpha: float, init_scaling_factor:float, random_seed:int):
        super().__init__()
        # Freeze the original layer's weights
        # Extract in_features and out_features from the shape tuple
        in_features, out_features = original_layer_shape
        self.original_layer = torch.nn.Linear(in_features, out_features)
        self.original_layer.requires_grad_(False)

        # Initialize the low-rank matrices A and B
        g = torch.Generator()
        g.manual_seed(random_seed)
        self.lora_A = nn.Parameter(torch.randn(in_features, rank, generator=g) * init_scaling_factor)
        print("lora A", self.lora_A)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        print("lora B", self.lora_B)

        # The scaling factor
        self.scaling = alpha / rank

    def forward(self, x):
        # Original output
        x = torch.tensor(x)
        original_output = self.original_layer(x)

        # LoRA update
        # (x @ self.lora_A) performs matrix multiplication of input x with matrix A
        # (x @ self.lora_A @ self.lora_B) is the full low-rank update
        lora_update = x @ self.lora_A @ self.lora_B

        # Final output is the sum of the original and the scaled LoRA update
        return original_output + lora_update * self.scaling