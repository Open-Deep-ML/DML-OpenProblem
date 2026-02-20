#!/usr/bin/env python
"""
convert_single_question.py
──────────────────────────
Paste ONE question dict into QUESTION_DICT and run:

    python utils/convert_single_question.py

The script splits it into:

questions/<id>_<slug>/
    ├─ meta.json
    ├─ description.md
    ├─ learn.md
    ├─ starter_code.py
    ├─ solution.py
    ├─ example.json
    ├─ tests.json
    ├─ tinygrad/   (optional)
    └─ pytorch/    (optional)
"""

import base64
import json
import pathlib
import re
from typing import Any, Dict

# ── 1️⃣  EDIT YOUR QUESTION HERE ────────────────────────────────────────────
QUESTION_DICT: Dict[str, Any] ={
    "description": "In this task, you will train a Generative Adversarial Network (GAN) to learn a one-dimensional Gaussian distribution. The GAN consists of a generator that produces samples from latent noise and a discriminator that estimates the probability that a given sample is real. Both networks should have one hidden layer with ReLU activation in the hidden layer. The generator’s output layer is linear, while the discriminator's output layer uses a sigmoid activation.\n\nYou must train the GAN using the standard non-saturating GAN loss for the generator and binary cross-entropy loss for the discriminator. In the NumPy version, parameters should be updated using vanilla gradient descent. In the PyTorch version, parameters should be updated using stochastic gradient descent (SGD) with the specified learning rate. The training loop should alternate between updating the discriminator and the generator each iteration.\n\nYour function must return the trained generator forward function `gen_forward(z)`, which produces generated samples given latent noise.",
    "id": "174",
    "test_cases": [
        {
            "test": "gen_forward = train_gan(4.0, 1.25, epochs=1000, seed=42)\nz = np.random.normal(0, 1, (500, 1))\nx_gen, _, _ = gen_forward(z)\nprint((round(np.mean(x_gen), 4), round(np.std(x_gen), 4)))",
            "expected_output": "(0.0004, 0.0002)"
        },
        {
            "test": "gen_forward = train_gan(0.0, 1.0, epochs=500, seed=0)\nz = np.random.normal(0, 1, (300, 1))\nx_gen, _, _ = gen_forward(z)\nprint((round(np.mean(x_gen), 4), round(np.std(x_gen), 4)))",
            "expected_output": "(-0.0002, 0.0002)"
        },
        {
            "test": "gen_forward = train_gan(-2.0, 0.5, epochs=1500, seed=123)\nz = np.random.normal(0, 1, (400, 1))\nx_gen, _, _ = gen_forward(z)\nprint((round(np.mean(x_gen), 4), round(np.std(x_gen), 4)))",
            "expected_output": "(-0.0044, 0.0002)"
        }
    ],
    "solution": "import numpy as np\n\ndef relu(x):\n    return np.maximum(0, x)\n\ndef sigmoid(x):\n    return 1 / (1 + np.exp(-x))\n\ndef train_gan(mean_real: float, std_real: float, latent_dim: int = 1, hidden_dim: int = 16, learning_rate: float = 0.001, epochs: int = 5000, batch_size: int = 128, seed: int = 42):\n    np.random.seed(seed)\n    data_dim = 1\n\n    # Initialize generator weights\n    w1_g = np.random.normal(0, 0.01, (latent_dim, hidden_dim))\n    b1_g = np.zeros(hidden_dim)\n    w2_g = np.random.normal(0, 0.01, (hidden_dim, data_dim))\n    b2_g = np.zeros(data_dim)\n\n    # Initialize discriminator weights\n    w1_d = np.random.normal(0, 0.01, (data_dim, hidden_dim))\n    b1_d = np.zeros(hidden_dim)\n    w2_d = np.random.normal(0, 0.01, (hidden_dim, 1))\n    b2_d = np.zeros(1)\n\n    def disc_forward(x):\n        h1 = np.dot(x, w1_d) + b1_d\n        a1 = relu(h1)\n        logit = np.dot(a1, w2_d) + b2_d\n        p = sigmoid(logit)\n        return p, logit, a1, h1\n\n    def gen_forward(z):\n        h1 = np.dot(z, w1_g) + b1_g\n        a1 = relu(h1)\n        x_gen = np.dot(a1, w2_g) + b2_g\n        return x_gen, a1, h1\n\n    for epoch in range(epochs):\n        # Sample real data\n        x_real = np.random.normal(mean_real, std_real, batch_size)[:, None]\n        z = np.random.normal(0, 1, (batch_size, latent_dim))\n        x_fake, _, _ = gen_forward(z)\n\n        # Discriminator forward\n        p_real, _, a1_real, h1_real = disc_forward(x_real)\n        p_fake, _, a1_fake, h1_fake = disc_forward(x_fake)\n\n        # Discriminator gradients\n        grad_logit_real = - (1 - p_real) / batch_size\n        grad_a1_real = grad_logit_real @ w2_d.T\n        grad_h1_real = grad_a1_real * (h1_real > 0)\n        grad_w1_d_real = x_real.T @ grad_h1_real\n        grad_b1_d_real = np.sum(grad_h1_real, axis=0)\n        grad_w2_d_real = a1_real.T @ grad_logit_real\n        grad_b2_d_real = np.sum(grad_logit_real, axis=0)\n\n        grad_logit_fake = p_fake / batch_size\n        grad_a1_fake = grad_logit_fake @ w2_d.T\n        grad_h1_fake = grad_a1_fake * (h1_fake > 0)\n        grad_w1_d_fake = x_fake.T @ grad_h1_fake\n        grad_b1_d_fake = np.sum(grad_h1_fake, axis=0)\n        grad_w2_d_fake = a1_fake.T @ grad_logit_fake\n        grad_b2_d_fake = np.sum(grad_logit_fake, axis=0)\n\n        grad_w1_d = grad_w1_d_real + grad_w1_d_fake\n        grad_b1_d = grad_b1_d_real + grad_b1_d_fake\n        grad_w2_d = grad_w2_d_real + grad_w2_d_fake\n        grad_b2_d = grad_b2_d_real + grad_b2_d_fake\n\n        w1_d -= learning_rate * grad_w1_d\n        b1_d -= learning_rate * grad_b1_d\n        w2_d -= learning_rate * grad_w2_d\n        b2_d -= learning_rate * grad_b2_d\n\n        # Generator update\n        z = np.random.normal(0, 1, (batch_size, latent_dim))\n        x_fake, a1_g, h1_g = gen_forward(z)\n        p_fake, _, a1_d, h1_d = disc_forward(x_fake)\n\n        grad_logit_fake = - (1 - p_fake) / batch_size\n        grad_a1_d = grad_logit_fake @ w2_d.T\n        grad_h1_d = grad_a1_d * (h1_d > 0)\n        grad_x_fake = grad_h1_d @ w1_d.T\n\n        grad_a1_g = grad_x_fake @ w2_g.T\n        grad_h1_g = grad_a1_g * (h1_g > 0)\n        grad_w1_g = z.T @ grad_h1_g\n        grad_b1_g = np.sum(grad_h1_g, axis=0)\n        grad_w2_g = a1_g.T @ grad_x_fake\n        grad_b2_g = np.sum(grad_x_fake, axis=0)\n\n        w1_g -= learning_rate * grad_w1_g\n        b1_g -= learning_rate * grad_b1_g\n        w2_g -= learning_rate * grad_w2_g\n        b2_g -= learning_rate * grad_b2_g\n\n    return gen_forward",
    "difficulty": "hard",
    "pytorch_difficulty": "medium",
    "video": "",
    "likes": "0",
    "dislikes": "0",
    "example": {
        "input": "gen_forward = train_gan(4.0, 1.25, epochs=1000, seed=42)\nz = np.random.normal(0, 1, (500, 1))\nx_gen, _, _ = gen_forward(z)\n(round(np.mean(x_gen), 4), round(np.std(x_gen), 4))",
        "output": "(0.0004, 0.0002)",
        "reasoning": "The test cases call `gen_forward` after training, sample 500 points, and then compute the mean and std."
    },
    "category": "Deep Learning",
    "pytorch_starter_code": "aW1wb3J0IHRvcmNoCmltcG9ydCB0b3JjaC5ubiBhcyBubgppbXBvcnQgdG9yY2gub3B0aW0gYXMgb3B0aW0KCmRlZiB0cmFpbl9nYW4obWVhbl9yZWFsOiBmbG9hdCwgc3RkX3JlYWw6IGZsb2F0LCBsYXRlbnRfZGltOiBpbnQgPSAxLCBoaWRkZW5fZGltOiBpbnQgPSAxNiwgbGVhcm5pbmdfcmF0ZTogZmxvYXQgPSAwLjAwMSwgZXBvY2hzOiBpbnQgPSA1MDAwLCBiYXRjaF9zaXplOiBpbnQgPSAxMjgsIHNlZWQ6IGludCA9IDQyKToKICAgIHRvcmNoLm1hbnVhbF9zZWVkKHNlZWQpCiAgICAjIFlvdXIgUHlUb3JjaCBpbXBsZW1lbnRhdGlvbiBoZXJlCiAgICBwYXNz",
    "title": "Train a Simple GAN on 1D Gaussian Data",
    "createdAt": "August 13, 2025 at 11:01:53 AM UTUTC-4",
    "contributor": [
        {
            "profile_link": "https://github.com/moe18",
            "name": "moe"
        }
    ],
    "pytorch_test_cases": [
        {
            "test": "gen_forward = train_gan(4.0, 1.25, epochs=100, seed=42)\nz = torch.randn(500, 1)\nx_gen = gen_forward(z)\nprint((round(x_gen.mean().item(), 4), round(x_gen.std().item(), 4)))",
            "expected_output": "(0.4725, 0.3563)"
        },
        {
            "test": "gen_forward = train_gan(0.0, 1.0, epochs=50, seed=0)\nz = torch.randn(300, 1)\nx_gen = gen_forward(z)\nprint((round(x_gen.mean().item(), 4), round(x_gen.std().item(), 4)))",
            "expected_output": "(0.0644, 0.244)"
        }
    ],
    "learn_section": "## Understanding GANs for 1D Gaussian Data\nA Generative Adversarial Network (GAN) consists of two neural networks - a **Generator** $G_\\theta$ and a **Discriminator** $D_\\phi$ - trained in a minimax game.\n\n### 1. The Roles\n- **Generator** $G_\\theta(z)$: Takes a latent noise vector $z \\sim \\mathcal{N}(0, I)$ and outputs a sample intended to resemble the real data.\n- **Discriminator** $D_\\phi(x)$: Outputs a probability $p \\in (0, 1)$ that the input $x$ came from the real data distribution rather than the generator.\n\n### 2. The Objective\nThe classical GAN objective is:\n$$\n\\min_{\\theta} \\; \\max_{\\phi} \\; \\mathbb{E}_{x \\sim p_{\\text{data}}} [\\log D_\\phi(x)] + \\mathbb{E}_{z \\sim p(z)} [\\log (1 - D_\\phi(G_\\theta(z)))]\n$$\nHere:\n- $p_{\\text{data}}$ is the real data distribution.\n- $p(z)$ is the prior distribution for the latent noise (often standard normal).\n\n### 3. Practical Losses\nIn implementation, we minimize:\n- **Discriminator loss**:\n$$\n\\mathcal{L}_D = - \\left( \\frac{1}{m} \\sum_{i=1}^m \\log D(x^{(i)}_{\\text{real}}) + \\log(1 - D(x^{(i)}_{\\text{fake}})) \\right)\n$$\n- **Generator loss** (non-saturating form):\n$$\n\\mathcal{L}_G = - \\frac{1}{m} \\sum_{i=1}^m \\log D(G(z^{(i)}))\n$$\n\n### 4. Forward/Backward Flow\n1. **Discriminator step**: Real samples $x_{\\text{real}}$ and fake samples $x_{\\text{fake}} = G(z)$ are passed through $D$, and $\\mathcal{L}_D$ is minimized w.r.t. $\\phi$.\n2. **Generator step**: Fresh $z$ is sampled, $x_{\\text{fake}} = G(z)$ is passed through $D$, and $\\mathcal{L}_G$ is minimized w.r.t. $\\theta$ while keeping $\\phi$ fixed.\n\n### 5. Architecture for This Task\n- **Generator**: Fully connected layer ($\\mathbb{R}^{\\text{latent\\_dim}} \\to \\mathbb{R}^{\\text{hidden\\_dim}}$) -> ReLU -> Fully connected layer ($\\mathbb{R}^{\\text{hidden\\_dim}} \\to \\mathbb{R}^1$).\n- **Discriminator**: Fully connected layer ($\\mathbb{R}^1 \\to \\mathbb{R}^{\\text{hidden\\_dim}}$) → ReLU → Fully connected layer ($\\mathbb{R}^{\\text{hidden\\_dim}} \\to \\mathbb{R}^1$) → Sigmoid.\n\n### 6. Numerical Tips\n- Initialize weights with a small Gaussian ($\\mathcal{N}(0, 0.01)$).\n- Add $10^{-8}$ to logs for numerical stability.\n- Use a consistent batch size $m$ for both real and fake samples.\n- Always sample fresh noise for the generator on each update.\n\n**Your Task**: Implement the training loop to learn the parameters $\\theta$ and $\\phi$, and return the trained `gen_forward(z)` function. The evaluation (mean/std of generated samples) will be handled in the test cases.",
    "pytorch_solution": "aW1wb3J0IHRvcmNoCmltcG9ydCB0b3JjaC5ubiBhcyBubgppbXBvcnQgdG9yY2gub3B0aW0gYXMgb3B0aW0KCmRlZiB0cmFpbl9nYW4obWVhbl9yZWFsOiBmbG9hdCwgc3RkX3JlYWw6IGZsb2F0LCBsYXRlbnRfZGltOiBpbnQgPSAxLCBoaWRkZW5fZGltOiBpbnQgPSAxNiwgbGVhcm5pbmdfcmF0ZTogZmxvYXQgPSAwLjAwMSwgZXBvY2hzOiBpbnQgPSA1MDAwLCBiYXRjaF9zaXplOiBpbnQgPSAxMjgsIHNlZWQ6IGludCA9IDQyKToKICAgIHRvcmNoLm1hbnVhbF9zZWVkKHNlZWQpCgogICAgY2xhc3MgR2VuZXJhdG9yKG5uLk1vZHVsZSk6CiAgICAgICAgZGVmIF9faW5pdF9fKHNlbGYpOgogICAgICAgICAgICBzdXBlcigpLl9faW5pdF9fKCkKICAgICAgICAgICAgc2VsZi5uZXQgPSBubi5TZXF1ZW50aWFsKAogICAgICAgICAgICAgICAgbm4uTGluZWFyKGxhdGVudF9kaW0sIGhpZGRlbl9kaW0pLAogICAgICAgICAgICAgICAgbm4uUmVMVSgpLAogICAgICAgICAgICAgICAgbm4uTGluZWFyKGhpZGRlbl9kaW0sIDEpCiAgICAgICAgICAgICkKICAgICAgICBkZWYgZm9yd2FyZChzZWxmLCB6KToKICAgICAgICAgICAgcmV0dXJuIHNlbGYubmV0KHopCgogICAgY2xhc3MgRGlzY3JpbWluYXRvcihubi5Nb2R1bGUpOgogICAgICAgIGRlZiBfX2luaXRfXyhzZWxmKToKICAgICAgICAgICAgc3VwZXIoKS5fX2luaXRfXygpCiAgICAgICAgICAgIHNlbGYubmV0ID0gbm4uU2VxdWVudGlhbCgKICAgICAgICAgICAgICAgIG5uLkxpbmVhcigxLCBoaWRkZW5fZGltKSwKICAgICAgICAgICAgICAgIG5uLlJlTFUoKSwKICAgICAgICAgICAgICAgIG5uLkxpbmVhcihoaWRkZW5fZGltLCAxKSwKICAgICAgICAgICAgICAgIG5uLlNpZ21vaWQoKQogICAgICAgICAgICApCiAgICAgICAgZGVmIGZvcndhcmQoc2VsZiwgeCk6CiAgICAgICAgICAgIHJldHVybiBzZWxmLm5ldCh4KQoKICAgIEcgPSBHZW5lcmF0b3IoKQogICAgRCA9IERpc2NyaW1pbmF0b3IoKQoKICAgICMgVXNlIFNHRCBhcyByZXF1ZXN0ZWQKICAgIG9wdF9HID0gb3B0aW0uU0dEKEcucGFyYW1ldGVycygpLCBscj1sZWFybmluZ19yYXRlKQogICAgb3B0X0QgPSBvcHRpbS5TR0QoRC5wYXJhbWV0ZXJzKCksIGxyPWxlYXJuaW5nX3JhdGUpCiAgICBjcml0ZXJpb24gPSBubi5CQ0VMb3NzKCkKCiAgICBmb3IgXyBpbiByYW5nZShlcG9jaHMpOgogICAgICAgICMgUmVhbCBhbmQgZmFrZSBiYXRjaGVzCiAgICAgICAgcmVhbF9kYXRhID0gdG9yY2gubm9ybWFsKG1lYW5fcmVhbCwgc3RkX3JlYWwsIHNpemU9KGJhdGNoX3NpemUsIDEpKQogICAgICAgIG5vaXNlID0gdG9yY2gucmFuZG4oYmF0Y2hfc2l6ZSwgbGF0ZW50X2RpbSkKICAgICAgICBmYWtlX2RhdGEgPSBHKG5vaXNlKQoKICAgICAgICAjIC0tLS0tIERpc2NyaW1pbmF0b3Igc3RlcCAtLS0tLQogICAgICAgIG9wdF9ELnplcm9fZ3JhZCgpCiAgICAgICAgcHJlZF9yZWFsID0gRChyZWFsX2RhdGEpCiAgICAgICAgcHJlZF9mYWtlID0gRChmYWtlX2RhdGEuZGV0YWNoKCkpCiAgICAgICAgbG9zc19yZWFsID0gY3JpdGVyaW9uKHByZWRfcmVhbCwgdG9yY2gub25lc19saWtlKHByZWRfcmVhbCkpCiAgICAgICAgbG9zc19mYWtlID0gY3JpdGVyaW9uKHByZWRfZmFrZSwgdG9yY2guemVyb3NfbGlrZShwcmVkX2Zha2UpKQogICAgICAgIGxvc3NfRCA9IGxvc3NfcmVhbCArIGxvc3NfZmFrZQogICAgICAgIGxvc3NfRC5iYWNrd2FyZCgpCiAgICAgICAgb3B0X0Quc3RlcCgpCgogICAgICAgICMgLS0tLS0gR2VuZXJhdG9yIHN0ZXAgLS0tLS0KICAgICAgICBvcHRfRy56ZXJvX2dyYWQoKQogICAgICAgIHByZWRfZmFrZSA9IEQoZmFrZV9kYXRhKQogICAgICAgICMgbm9uLXNhdHVyYXRpbmcgZ2VuZXJhdG9yIGxvc3M6IG1heGltaXplIGxvZyBEKEcoeikpIC0+IG1pbmltaXplIC1sb2cgRChHKHopKQogICAgICAgIGxvc3NfRyA9IGNyaXRlcmlvbihwcmVkX2Zha2UsIHRvcmNoLm9uZXNfbGlrZShwcmVkX2Zha2UpKQogICAgICAgIGxvc3NfRy5iYWNrd2FyZCgpCiAgICAgICAgb3B0X0cuc3RlcCgpCgogICAgcmV0dXJuIEcuZm9yd2FyZA==",
    "starter_code": "import numpy as np\n\ndef train_gan(mean_real: float, std_real: float, latent_dim: int = 1, hidden_dim: int = 16, learning_rate: float = 0.001, epochs: int = 5000, batch_size: int = 128, seed: int = 42):\n    \"\"\"\n    Train a simple GAN to learn a 1D Gaussian distribution.\n\n    Args:\n        mean_real: Mean of the target Gaussian\n        std_real: Std of the target Gaussian\n        latent_dim: Dimension of the noise input to the generator\n        hidden_dim: Hidden layer size for both networks\n        learning_rate: Learning rate for gradient descent\n        epochs: Number of training epochs\n        batch_size: Training batch size\n        seed: Random seed for reproducibility\n\n    Returns:\n        gen_forward: A function that takes z and returns generated samples\n    \"\"\"\n    # Your code here\n    pass"
}




# ────────────────────────────────────────────────────────────────────────────


# ---------- helpers ---------------------------------------------------------
def slugify(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "-", text.lower())
    return re.sub(r"-{2,}", "-", text).strip("-")[:50]


def maybe_b64(s: str) -> str:
    try:
        if len(s) % 4 == 0 and re.fullmatch(r"[0-9A-Za-z+/=\n\r]+", s):
            return base64.b64decode(s).decode("utf-8")
    except Exception:
        pass
    return s


def write_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip("\n") + "\n", encoding="utf-8")


def write_json(path: pathlib.Path, obj: Any) -> None:
    write_text(path, json.dumps(obj, indent=2, ensure_ascii=False))


# ---------- converter -------------------------------------------------------
def convert_one(q: Dict[str, Any]) -> None:
    folder = pathlib.Path("questions") / f"{q['id']}_{slugify(q['title'])}"
    folder.mkdir(parents=True, exist_ok=True)

    # meta.json
    meta = {
        "id": q["id"],
        "title": q["title"],
        "difficulty": q["difficulty"],
        "category": q["category"],
        "video": q.get("video", ""),
        "likes": q.get("likes", "0"),
        "dislikes": q.get("dislikes", "0"),
        "contributor": q.get("contributor", []),
    }
    for opt in ("tinygrad_difficulty", "pytorch_difficulty", "marimo_link"):
        if opt in q:
            meta[opt] = q[opt]
    write_json(folder / "meta.json", meta)

    # core files
    write_text(folder / "description.md", q["description"])
    write_text(folder / "learn.md", q["learn_section"])
    write_text(folder / "starter_code.py", q["starter_code"])
    write_text(folder / "solution.py", q["solution"])
    write_json(folder / "example.json", q["example"])
    write_json(folder / "tests.json", q["test_cases"])

    # optional language-specific extras
    for lang in ("tinygrad", "pytorch"):
        sc, so, tc = (f"{lang}_starter_code", f"{lang}_solution", f"{lang}_test_cases")
        if any(k in q for k in (sc, so, tc)):
            sub = folder / lang
            if sc in q:
                write_text(sub / "starter_code.py", maybe_b64(q[sc]))
            if so in q:
                write_text(sub / "solution.py", maybe_b64(q[so]))
            if tc in q:
                write_json(sub / "tests.json", q[tc])

    # success message (relative if possible)
    try:
        rel = folder.relative_to(pathlib.Path.cwd())
    except ValueError:
        rel = folder
    print(f"✓  Created {rel}")


# ---------- main ------------------------------------------------------------
def main() -> None:
    convert_one(QUESTION_DICT)


if __name__ == "__main__":
    main()
