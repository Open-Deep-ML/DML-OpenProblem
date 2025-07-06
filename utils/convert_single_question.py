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
  "title": "Incremental Mean for Online Reward Estimation",
  "description": "Implement an efficient method to update the mean reward for a k-armed bandit action after receiving each new reward, **without storing the full history of rewards**. Given the previous mean estimate (Q_prev), the number of times the action has been selected (k), and a new reward (R), compute the updated mean using the incremental formula.\n\n**Note:** Using a regular mean that stores all past rewards will eventually run out of memory. Your solution should use only the previous mean, the count, and the new reward.",
  "category": "Reinforcement Learning",
  "difficulty": "easy",
  "starter_code": "def incremental_mean(Q_prev, k, R):\n    \"\"\"\n    Q_prev: previous mean estimate (float)\n    k: number of times the action has been selected (int)\n    R: new observed reward (float)\n    Returns: new mean estimate (float)\n    \"\"\"\n    # Your code here\n    pass\n",
  "solution": "def incremental_mean(Q_prev, k, R):\n    return Q_prev + (1 / k) * (R - Q_prev)",
  "test_cases": [
    {
      "test": "Q = 0.0\nk = 1\nR = 5.0\nprint(round(incremental_mean(Q, k, R), 4))",
      "expected_output": "5.0"
    },
    {
      "test": "Q = 5.0\nk = 2\nR = 7.0\nprint(round(incremental_mean(Q, k, R), 4))",
      "expected_output": "6.0"
    },
    {
      "test": "Q = 6.0\nk = 3\nR = 4.0\nprint(round(incremental_mean(Q, k, R), 4))",
      "expected_output": "5.3333"
    }
  ],
  "example": {
    "input": "Q_prev = 2.0\nk = 2\nR = 6.0\nnew_Q = incremental_mean(Q_prev, k, R)\nprint(round(new_Q, 2))",
    "output": "4.0",
    "reasoning": "The updated mean is Q_prev + (1/k) * (R - Q_prev) = 2.0 + (1/2)*(6.0 - 2.0) = 2.0 + 2.0 = 4.0"
  },
  "learn_section": "### Incremental Mean Update Rule\n\nThe incremental mean formula lets you update your estimate of the mean after each new observation, **without keeping all previous rewards in memory**. For the k-th reward $R_k$ and previous estimate $Q_{k}$:\n\n$$\nQ_{k+1} = Q_k + \\frac{1}{k} (R_k - Q_k)\n$$\n\nThis saves memory compared to the regular mean, which requires storing all past rewards and recalculating each time. The incremental rule is crucial for online learning and large-scale problems where storing all data is impractical."
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
