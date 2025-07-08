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
  "id":'158',
  "title": "Epsilon-Greedy Action Selection for n-Armed Bandit",
  "description": "Implement the epsilon-greedy method for action selection in an n-armed bandit problem. Given a set of estimated action values (Q-values), select an action using the epsilon-greedy policy: with probability epsilon, choose a random action; with probability 1 - epsilon, choose the action with the highest estimated value.",
  "category": "Reinforcement Learning",
  "difficulty": "easy",
  "starter_code": "import numpy as np\n\ndef epsilon_greedy(Q, epsilon=0.1):\n    \"\"\"\n    Selects an action using epsilon-greedy policy.\n    Q: np.ndarray of shape (n,) -- estimated action values\n    epsilon: float in [0, 1]\n    Returns: int, selected action index\n    \"\"\"\n    # Your code here\n    pass",
  "solution": "import numpy as np\n\ndef epsilon_greedy(Q, epsilon=0.1):\n    if np.random.rand() < epsilon:\n        return np.random.randint(len(Q))\n    else:\n        return int(np.argmax(Q))",
  "test_cases": [
    {
      "test": "import numpy as np\nnp.random.seed(0)\nprint([epsilon_greedy(np.array([1, 2, 3]), epsilon=0.0) for _ in range(5)])",
      "expected_output": "[2, 2, 2, 2, 2]"
    },
    {
      "test": "import numpy as np\nnp.random.seed(1)\nprint([epsilon_greedy(np.array([5, 2, 1]), epsilon=1.0) for _ in range(5)])",
      "expected_output": "[0, 1, 1, 0, 0]"
    },
    {
      "test": "import numpy as np\nnp.random.seed(42)\nresults = [epsilon_greedy(np.array([1.5, 2.5, 0.5]), epsilon=0.5) for _ in range(10)]\nprint(results)",
      "expected_output": "[1, 0, 1, 1, 1, 0, 1, 0, 0, 0]"
    }
  ],
  "example": {
    "input": "Q = np.array([0.5, 2.3, 1.7])\nepsilon = 0.0\naction = epsilon_greedy(Q, epsilon)\nprint(action)",
    "output": "1",
    "reasoning": "With epsilon=0.0 (always greedy), the highest Q-value is 2.3 at index 1, so the function always returns 1."
  },
  "learn_section": "### Epsilon-Greedy Policy\n\nThe epsilon-greedy method is a fundamental action selection strategy used in reinforcement learning, especially for solving the n-armed bandit problem. The key idea is to balance **exploration** (trying new actions) and **exploitation** (choosing the best-known action):\n\n- With probability $\\varepsilon$ (epsilon), the agent explores by selecting an action at random.\n- With probability $1-\\varepsilon$, it exploits by choosing the action with the highest estimated value (greedy choice).\n\nThe epsilon-greedy policy is simple to implement and provides a way to avoid getting stuck with suboptimal actions due to insufficient exploration."
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
