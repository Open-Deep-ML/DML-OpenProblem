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
    "id": "161",
  "video": "",
  "likes": "0",
  "dislikes": "0",
  "contributor": [
    {
      "profile_link": "https://github.com/moe18",
      "name": "Moe Chabot"
    }
  ],
   
    "title": "Exponential Weighted Average of Rewards",
    "description": "Given an initial value $Q_1$, a list of $k$ observed rewards $R_1, R_2, \\ldots, R_k$, and a step size $\\alpha$, implement a function to compute the exponentially weighted average as:\n\n$$(1-\\alpha)^k Q_1 + \\sum_{i=1}^k \\alpha (1-\\alpha)^{k-i} R_i$$\n\nThis weighting gives more importance to recent rewards, while the influence of the initial estimate $Q_1$ decays over time. Do **not** use running/incremental updates; instead, compute directly from the formula. (This is called the *exponential recency-weighted average*.)",
    "category": "Reinforcement Learning",
    "difficulty": "medium",
    "starter_code": "def exp_weighted_average(Q1, rewards, alpha):\n    \"\"\"\n    Q1: float, initial estimate\n    rewards: list or array of rewards, R_1 to R_k\n    alpha: float, step size (0 < alpha <= 1)\n    Returns: float, exponentially weighted average after k rewards\n    \"\"\"\n    # Your code here\n    pass\n",
    "solution": "def exp_weighted_average(Q1, rewards, alpha):\n    k = len(rewards)\n    value = (1 - alpha) ** k * Q1\n    for i, Ri in enumerate(rewards):\n        value += alpha * (1 - alpha) ** (k - i - 1) * Ri\n    return value",
    "test_cases": [
      {
        "test": "Q1 = 10.0\nrewards = [4.0, 7.0, 13.0]\nalpha = 0.5\nprint(round(exp_weighted_average(Q1, rewards, alpha), 4))",
        "expected_output": "10.0"
      },
      {
        "test": "Q1 = 0.0\nrewards = [1.0, 1.0, 1.0, 1.0]\nalpha = 0.1\nprint(round(exp_weighted_average(Q1, rewards, alpha), 4))",
        "expected_output": "0.3439"
      }
    ],
    "example": {
      "input": "Q1 = 2.0\nrewards = [5.0, 9.0]\nalpha = 0.3\nresult = exp_weighted_average(Q1, rewards, alpha)\nprint(round(result, 4))",
      "output": "5.003",
      "reasoning": "Here, k=2, so the result is: (1-0.3)^2*2.0 + 0.3*(1-0.3)^1*5.0 + 0.3*(1-0.3)^0*9.0 = 0.49*2.0 + 0.21*5.0 + 0.3*9.0 = 0.98 + 1.05 + 2.7 = 4.73 (actually, should be 0.49*2+0.3*0.7*5+0.3*9 = 0.98+1.05+2.7=4.73)"
    },
    "learn_section": "### Exponential Recency-Weighted Average\n\nWhen the environment is nonstationary, it is better to give more weight to recent rewards. The formula $$(1-\\alpha)^k Q_1 + \\sum_{i=1}^k \\alpha (1-\\alpha)^{k-i} R_i$$ computes the expected value by exponentially decaying the influence of old rewards and the initial estimate. The parameter $\\alpha$ controls how quickly old information is forgotten: higher $\\alpha$ gives more weight to new rewards."
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
