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
    "id": "142",
    "title": "Gridworld Policy Evaluation",
    "description": "Implement policy evaluation for a 5x5 gridworld. Given a policy (mapping each state to action probabilities), compute the state-value function $V(s)$ for each cell using the Bellman expectation equation. The agent can move up, down, left, or right, receiving a constant reward of -1 for each move. Terminal states (the four corners) are fixed at 0. Iterate until the largest change in $V$ is less than a given threshold. Only use Python built-ins and no external RL libraries.",
    "test_cases": [
        {
            "test": "grid_size = 5\ngamma = 0.9\nthreshold = 0.001\npolicy = {(i, j): {'up': 0.25, 'down': 0.25, 'left': 0.25, 'right': 0.25} for i in range(grid_size) for j in range(grid_size)}\nV = gridworld_policy_evaluation(policy, gamma, threshold)\nprint([round(V[2][2], 4), V[0][0], V[0][4], V[4][0], V[4][4]])",
            "expected_output": "[-7.0902, 0.0, 0.0, 0.0, 0.0]"
        },
        {
            "test": "grid_size = 5\ngamma = 0.9\nthreshold = 0.001\npolicy = {(i, j): {'up': 0.1, 'down': 0.4, 'left': 0.1, 'right': 0.4} for i in range(grid_size) for j in range(grid_size)}\nV = gridworld_policy_evaluation(policy, gamma, threshold)\nprint(round(V[1][3], 4) < 0)",
            "expected_output": "True"
        }
    ],
    "solution": "def gridworld_policy_evaluation(policy: dict, gamma: float, threshold: float) -> list[list[float]]:\n    grid_size = 5\n    V = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]\n    actions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}\n    reward = -1\n    while True:\n        delta = 0.0\n        new_V = [row[:] for row in V]\n        for i in range(grid_size):\n            for j in range(grid_size):\n                if (i, j) in [(0, 0), (0, grid_size-1), (grid_size-1, 0), (grid_size-1, grid_size-1)]:\n                    continue\n                v = 0.0\n                for action, prob in policy[(i, j)].items():\n                    di, dj = actions[action]\n                    ni = i + di if 0 <= i + di < grid_size else i\n                    nj = j + dj if 0 <= j + dj < grid_size else j\n                    v += prob * (reward + gamma * V[ni][nj])\n                new_V[i][j] = v\n                delta = max(delta, abs(V[i][j] - new_V[i][j]))\n        V = new_V\n        if delta < threshold:\n            break\n    return V",
    "example": {
        "input": "policy = {(i, j): {'up': 0.25, 'down': 0.25, 'left': 0.25, 'right': 0.25} for i in range(5) for j in range(5)}\ngamma = 0.9\nthreshold = 0.001\nV = gridworld_policy_evaluation(policy, gamma, threshold)\nprint(round(V[2][2], 4))",
        "output": "-7.0902",
        "reasoning": "The policy is uniform (equal chance of each move). The agent receives -1 per step. After iterative updates, the center state value converges to about -7.09, and corners remain at 0."
    },
    "category": "Reinforcement Learning",
    "starter_code": "def gridworld_policy_evaluation(policy: dict, gamma: float, threshold: float) -> list[list[float]]:\n    \"\"\"\n    Evaluate state-value function for a policy on a 5x5 gridworld.\n    \n    Args:\n        policy: dict mapping (row, col) to action probability dicts\n        gamma: discount factor\n        threshold: convergence threshold\n    Returns:\n        5x5 list of floats\n    \"\"\"\n    # Your code here\n    pass",
    "learn_section": r"""# Gridworld Policy Evaluation

In reinforcement learning, **policy evaluation** is the process of computing the state-value function for a given policy. For a gridworld environment, this involves iteratively updating the value of each state based on the expected return following the policy.

## Key Concepts

- **State-Value Function (V):**  
  The expected return when starting from a state and following a given policy.

- **Policy:**  
  A mapping from states to probabilities of selecting each available action.

- **Bellman Expectation Equation:**  
  For each state $s$:
  $$
  V(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V(s')]
  $$
  where:
  - $ \pi(a|s) $ is the probability of taking action $ a $ in state $ s $,
  - $ P(s'|s,a) $ is the probability of transitioning to state $ s' $,
  - $ R(s,a,s') $ is the reward for that transition,
  - $ \gamma $ is the discount factor.

## Algorithm Overview

1. **Initialization:**  
   Start with an initial guess (commonly zeros) for the state-value function $ V(s) $.

2. **Iterative Update:**  
   For each non-terminal state, update the state value using the Bellman expectation equation. Continue updating until the maximum change in value (delta) is less than a given threshold.

3. **Terminal States:**  
   For this example, the four corners of the grid are considered terminal, so their values remain unchanged.

This evaluation method is essential for understanding how "good" each state is under a specific policy, and it forms the basis for more advanced reinforcement learning algorithms.""",
    "contributor": [
        {
            "profile_link": "https://github.com/arpitsinghgautam",
            "name": "Arpit Singh Gautam"
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
