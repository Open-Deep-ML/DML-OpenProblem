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
QUESTION_DICT: Dict[str, Any] = {
    "id": "143",
    "title": "Instance Normalization (IN) Implementation",
    "description": "Implement the Instance Normalization operation for 4D tensors (B, C, H, W) using NumPy. For each instance in the batch and each channel, normalize the spatial dimensions (height and width) by subtracting the mean and dividing by the standard deviation, then apply a learned scale (gamma) and shift (beta).",
    "test_cases": [
        {
            "test": "import numpy as np\nB, C, H, W = 2, 2, 2, 2\nnp.random.seed(42)\nX = np.random.randn(B, C, H, W)\ngamma = np.ones(C)\nbeta = np.zeros(C)\nout = instance_normalization(X, gamma, beta)\nprint(np.round(out[1][1], 4))",
            "expected_output": "[[ 1.4005, -1.0503] [-0.8361, 0.486 ]]"
        },
        {
            "test": "import numpy as np\nB, C, H, W = 2, 2, 2, 2\nnp.random.seed(101)\nX = np.random.randn(B, C, H, W)\ngamma = np.ones(C)\nbeta = np.zeros(C)\nout = instance_normalization(X, gamma, beta)\nprint(np.round(out[1][0], 4))",
            "expected_output": "[[-1.537, 0.9811], [ 0.7882, -0.2323]]"
        },
        {
            "test": "import numpy as np\nB, C, H, W = 2, 2, 2, 2\nnp.random.seed(101)\nX = np.random.randn(B, C, H, W)\ngamma = np.ones(C) * 0.5\nbeta = np.ones(C)\nout = instance_normalization(X, gamma, beta)\nprint(np.round(out[0][0], 4))",
            "expected_output": "[[1.8542, 0.6861], [0.8434, 0.6163]]"
        }
    ],
    "solution": "import numpy as np\n\ndef instance_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:\n    # Reshape gamma, beta for broadcasting: (1, C, 1, 1)\n    gamma = gamma.reshape(1, -1, 1, 1)\n    beta = beta.reshape(1, -1, 1, 1)\n    mean = np.mean(X, axis=(2, 3), keepdims=True)\n    var = np.var(X, axis=(2, 3), keepdims=True)\n    X_norm = (X - mean) / np.sqrt(var + epsilon)\n    return gamma * X_norm + beta",
    "example": {
        "input": "import numpy as np\nB, C, H, W = 2, 2, 2, 2\nnp.random.seed(42)\nX = np.random.randn(B, C, H, W)\ngamma = np.ones(C)\nbeta = np.zeros(C)\nout = instance_normalization(X, gamma, beta)\nprint(np.round(out, 8))",
        "output": "[[[[-0.08841405 -0.50250083]\n   [ 0.01004046  0.58087442]]\n\n  [[-0.43833369 -0.43832346]\n   [ 0.69114093  0.18551622]]]\n\n [[[-0.17259136  0.51115219]\n   [-0.16849938 -0.17006144]]\n\n  [[ 0.73955155 -0.55463639]\n   [-0.44152783  0.25661268]]]]",
        "reasoning": "The function normalizes each instance and channel across (H, W), then applies the gamma and beta scaling/shifting parameters. This matches standard InstanceNorm behavior."
    },
    "category": "Deep Learning",
    "starter_code": "import numpy as np\n\ndef instance_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:\n    \"\"\"\n    Perform Instance Normalization over a 4D tensor X of shape (B, C, H, W).\n    gamma: scale parameter of shape (C,)\n    beta: shift parameter of shape (C,)\n    epsilon: small value for numerical stability\n    Returns: normalized array of same shape as X\n    \"\"\"\n    # TODO: Implement Instance Normalization\n    pass",
    "learn_section": r"""## Understanding Instance Normalization

Instance Normalization (IN) is a normalization technique primarily used in image generation and style transfer tasks. Unlike Batch Normalization or Group Normalization, Instance Normalization normalizes each individual sample (or instance) separately, across its spatial dimensions. This is particularly effective in applications like style transfer, where normalization is needed per image to preserve the content while allowing different styles to be applied.

### Concepts

Instance Normalization operates on the principle of normalizing each individual sample independently. This helps to remove the style information from the images, leaving only the content. By normalizing each instance, the method allows the model to focus on the content of the image rather than the variations between images in a batch.

The process of Instance Normalization consists of the following steps:

1. **Compute the Mean and Variance for Each Instance:** For each instance (image), compute the mean and variance across its spatial dimensions.
2. **Normalize the Inputs:** Normalize each instance using the computed mean and variance.
3. **Apply Scale and Shift:** After normalization, apply a learned scale (gamma) and shift (beta) to restore the model's ability to represent the data's original distribution.

### Structure of Instance Normalization for BCHW Input

For an input tensor with the shape **BCHW** , where:
- **B**: batch size,
- **C**: number of channels,
- **H**: height,
- **W**: width,
Instance Normalization operates on the spatial dimensions (height and width) of each instance (image) separately.

#### 1. Mean and Variance Calculation for Each Instance

- For each individual instance in the batch (for each **b** in **B**), the **mean** $\mu_b$ and **variance** $\sigma_b^2$ are computed across the spatial dimensions (height and width), but **independently for each channel**.

  $$ 
  \mu_b = \frac{1}{H \cdot W} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{b,c,h,w}
  $$

  $$
  \sigma_b^2 = \frac{1}{H \cdot W} \sum_{h=1}^{H} \sum_{w=1}^{W} (x_{b,c,h,w} - \mu_b)^2
  $$

  Where:
  - $x_{b,c,h,w}$ is the activation at batch index $b$, channel $c$, height $h$, and width $w$.
  - $H$ and $W$ are the spatial dimensions (height and width).

#### 2. Normalization

Once the mean $\mu_b$ and variance $\sigma_b^2$ have been computed for each instance, the next step is to **normalize** the input for each instance across the spatial dimensions (height and width), for each channel:

$$
\hat{x}_{b,c,h,w} = \frac{x_{b,c,h,w} - \mu_b}{\sqrt{\sigma_b^2 + \epsilon}}
$$

Where:
- $\hat{x}_{b,c,h,w}$ is the normalized activation for the input at batch index $b$, channel index $c$, height $h$, and width $w$.
- $\epsilon$ is a small constant added to the variance for numerical stability.

#### 3. Scale and Shift

After normalization, the next step is to apply a **scale** ($\gamma_c$) and **shift** ($\beta_c$) to the normalized activations for each channel. These learned parameters allow the model to adjust the output distribution for each channel:

$$
y_{b,c,h,w} = \gamma_c \hat{x}_{b,c,h,w} + \beta_c
$$

Where:
- $\gamma_c$ is the scaling factor for channel $c$.
- $\beta_c$ is the shifting factor for channel $c$.

#### 4. Training and Inference

- **During Training**: The mean and variance are computed for each instance in the mini-batch and used for normalization.
- **During Inference**: The model uses the running averages of the statistics (mean and variance) computed during training to ensure consistent behavior in production.

### Key Points

- **Instance-wise Normalization**: Instance Normalization normalizes each image independently, across its spatial dimensions (height and width) and across the channels.
  
- **Style Transfer**: This normalization technique is widely used in **style transfer** tasks, where each image must be normalized independently to allow for style information to be adjusted without affecting the content.

- **Batch Independence**: Instance Normalization does not depend on the batch size, as normalization is applied per instance, making it suitable for tasks where per-image normalization is critical.

- **Numerical Stability**: A small constant $\epsilon$ is added to the variance to avoid numerical instability when dividing by the square root of the variance.

- **Improved Training in Style-Related Tasks**: Instance Normalization helps to remove unwanted style-related variability across different images, allowing for better performance in tasks like style transfer, where the goal is to separate content and style information.

### Why Normalize Over Instances?

- **Content Preservation**: By normalizing each image individually, Instance Normalization allows the model to preserve the content of the images while adjusting the style. This makes it ideal for style transfer and other image manipulation tasks.
  
- **Batch Independence**: Unlike Batch Normalization, which requires large batch sizes to compute statistics, Instance Normalization normalizes each image independently, making it suitable for tasks where the batch size is small or varies.

- **Reducing Style Variability**: Instance Normalization removes the variability in style information across a batch, allowing for a consistent representation of content across different images.

In summary, Instance Normalization is effective for image-based tasks like style transfer, where the goal is to normalize each image independently to preserve its content while allowing style modifications.""",
    "contributor": [
        {
            "profile_link": "https://github.com/nzomi",
            "name": "nzomi"
        }
    ],
    "likes": "0",
    "dislikes": "0",
    "difficulty": "medium",
    "video": ""
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
