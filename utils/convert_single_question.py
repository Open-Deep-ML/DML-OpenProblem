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
    'id':'140',
    "description": "Write a Python class to implement the Bernoulli Naive Bayes classifier for binary (0/1) feature data. Your class should have two methods: `forward(self, X, y)` to train on the input data (X: 2D NumPy array of binary features, y: 1D NumPy array of class labels) and `predict(self, X)` to output predicted labels for a 2D test matrix X. Use Laplace smoothing (parameter: smoothing=1.0). Return predictions as a NumPy array. Only use NumPy. Predictions must be binary (0 or 1) and you must handle cases where the training data contains only one class. All log/likelihood calculations should use log probabilities for numerical stability.",
    "test_cases": [
        {
            "test": "import numpy as np\nmodel = NaiveBayes(smoothing=1.0)\nX = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1]])\ny = np.array([1, 1, 0, 0, 1])\nmodel.forward(X, y)\nprint(model.predict(np.array([[1, 0, 1]])))",
            "expected_output": "[1]"
        },
        {
            "test": "import numpy as np\nmodel = NaiveBayes(smoothing=1.0)\nX = np.array([[0], [1], [0], [1]])\ny = np.array([0, 1, 0, 1])\nmodel.forward(X, y)\nprint(model.predict(np.array([[0], [1]])))",
            "expected_output": "[0 1]"
        },
        {
            "test": "import numpy as np\nmodel = NaiveBayes(smoothing=1.0)\nX = np.array([[0, 0], [1, 0], [0, 1]])\ny = np.array([0, 1, 0])\nmodel.forward(X, y)\nprint(model.predict(np.array([[1, 1]])))",
            "expected_output": "[0]"
        },
        {
            "test": "import numpy as np\nnp.random.seed(42)\nmodel = NaiveBayes(smoothing=1.0)\nX = np.random.randint(0, 2, (100, 5))\ny = np.random.choice([0, 1], size=100)\nmodel.forward(X, y)\nX_test = np.random.randint(0, 2, (10, 5))\npred = model.predict(X_test)\nprint(pred.shape)",
            "expected_output": "(10,)"
        },
        {
            "test": "import numpy as np\nmodel = NaiveBayes(smoothing=1.0)\nX = np.random.randint(0, 2, (10, 3))\ny = np.zeros(10)\nmodel.forward(X, y)\nX_test = np.random.randint(0, 2, (3, 3))\nprint(model.predict(X_test))",
            "expected_output": "[0, 0, 0]"
        }
    ],
    "solution": "import numpy as np\n\nclass NaiveBayes():\n    def __init__(self, smoothing=1.0):\n        self.smoothing = smoothing\n        self.classes = None\n        self.priors = None\n        self.likelihoods = None\n\n    def forward(self, X, y):\n        self.classes, class_counts = np.unique(y, return_counts=True)\n        self.priors = {cls: np.log(class_counts[i] / len(y)) for i, cls in enumerate(self.classes)}\n        self.likelihoods = {}\n        for cls in self.classes:\n            X_cls = X[y == cls]\n            prob = (np.sum(X_cls, axis=0) + self.smoothing) / (X_cls.shape[0] + 2 * self.smoothing)\n            self.likelihoods[cls] = (np.log(prob), np.log(1 - prob))\n\n    def _compute_posterior(self, sample):\n        posteriors = {}\n        for cls in self.classes:\n            posterior = self.priors[cls]\n            prob_1, prob_0 = self.likelihoods[cls]\n            likelihood = np.sum(sample * prob_1 + (1 - sample) * prob_0)\n            posterior += likelihood\n            posteriors[cls] = posterior\n        return max(posteriors, key=posteriors.get)\n\n    def predict(self, X):\n        return np.array([self._compute_posterior(sample) for sample in X])",
    "example": {
        "input": "X = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1]]); y = np.array([1, 1, 0, 0, 1])\nmodel = NaiveBayes(smoothing=1.0)\nmodel.forward(X, y)\nprint(model.predict(np.array([[1, 0, 1]])))",
        "output": "[1]",
        "reasoning": "The model learns class priors and feature probabilities with Laplace smoothing. For [1, 0, 1], the posterior for class 1 is higher, so the model predicts 1."
    },
    "category": "Machine Learning",
    "starter_code": "import numpy as np\n\nclass NaiveBayes():\n    def __init__(self, smoothing=1.0):\n        # Initialize smoothing\n        pass\n\n    def forward(self, X, y):\n        # Fit model to binary features X and labels y\n        pass\n\n    def predict(self, X):\n        # Predict class labels for test set X\n        pass",
    "title": "Bernoulli Naive Bayes Classifier",
    "learn_section":r"""# **Naive Bayes Classifier**

## **1. Definition**

Naive Bayes is a **probabilistic machine learning algorithm** used for **classification tasks**. It is based on **Bayes' Theorem**, which describes the probability of an event based on prior knowledge of related events.

The algorithm assumes that:
- **Features are conditionally independent** given the class label (the "naive" assumption).
- It calculates the posterior probability for each class and assigns the class with the **highest posterior** to the sample.

---

## **2. Bayes' Theorem**

Bayes' Theorem is given by:

$$
P(C | X) = \frac{P(X | C) \times P(C)}{P(X)}
$$

Where:
- $P(C | X)$ **Posterior** probability: the probability of class $C $ given the feature vector $X$
- $P(X | C)$ → **Likelihood**: the probability of the data $X$ given the class
- $P(C)$ → **Prior** probability: the initial probability of class $C$ before observing any data
- $ P(X)$ → **Evidence**: the total probability of the data across all classes (acts as a normalizing constant)

Since $P(X)$ is the same for all classes during comparison, it can be ignored, simplifying the formula to:

$$
P(C | X) \propto P(X | C) \times P(C)
$$
---

### 3 **Bernoulli Naive Bayes**
- Used for **binary data** (features take only 0 or 1 values).
- The likelihood is given by:

$$
P(X | C) = \prod_{i=1}^{n} P(x_i | C)^{x_i} \cdot (1 - P(x_i | C))^{1 - x_i}
$$

---

## **4. Applications of Naive Bayes**

- **Text Classification:** Spam detection, sentiment analysis, and news categorization.
- **Document Categorization:** Sorting documents by topic.
- **Fraud Detection:** Identifying fraudulent transactions or behaviors.
- **Recommender Systems:** Classifying users into preference groups.

--- """,
    "contributor": [
        {
            "profile_link": "https://github.com/moe18",
            "name": "Moe Chabot"
        }
    ],
    "likes": "0",
    "dislikes": "0",
    "difficulty": "medium",
    "video":''
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
