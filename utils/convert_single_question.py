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
  "id": "157",
  "title": "Implement the Bellman Equation for Value Iteration",
  "description": "Write a function that performs one step of value iteration for a given Markov Decision Process (MDP) using the Bellman equation. The function should update the state-value function V(s) for each state based on possible actions, transition probabilities, rewards, and the discount factor gamma. Only use NumPy.",
  "test_cases": [
    {
      "test": "import numpy as np\ntransitions = [\n  # For state 0\n  {0: [(1.0, 0, 0.0, False)], 1: [(1.0, 1, 1.0, False)]},\n  # For state 1\n  {0: [(1.0, 0, 0.0, False)], 1: [(1.0, 1, 1.0, True)]}\n]\nV = np.array([0.0, 0.0])\ngamma = 0.9\nnew_V = bellman_update(V, transitions, gamma)\nprint(np.round(new_V, 2))",
      "expected_output": "[1., 1.]"
    },
    {
      "test": "import numpy as np\ntransitions = [\n  {0: [(0.8, 0, 5, False), (0.2, 1, 10, False)], 1: [(1.0, 1, 2, False)]},\n  {0: [(1.0, 0, 0, False)], 1: [(1.0, 1, 0, True)]}\n]\nV = np.array([0.0, 0.0])\ngamma = 0.5\nnew_V = bellman_update(V, transitions, gamma)\nprint(np.round(new_V, 2))",
      "expected_output": "[6.,  0.]"
    }
  ],
  "solution": "import numpy as np\n\ndef bellman_update(V, transitions, gamma):\n    n_states = len(V)\n    new_V = np.zeros_like(V)\n    for s in range(n_states):\n        action_values = []\n        for a in transitions[s]:\n            total = 0\n            for prob, next_s, reward, done in transitions[s][a]:\n                total += prob * (reward + gamma * (0 if done else V[next_s]))\n            action_values.append(total)\n        new_V[s] = max(action_values)\n    return new_V",
  "example": {
    "input": "import numpy as np\ntransitions = [\n  {0: [(1.0, 0, 0.0, False)], 1: [(1.0, 1, 1.0, False)]},\n  {0: [(1.0, 0, 0.0, False)], 1: [(1.0, 1, 1.0, True)]}\n]\nV = np.array([0.0, 0.0])\ngamma = 0.9\nnew_V = bellman_update(V, transitions, gamma)\nprint(np.round(new_V, 2))",
    "output": "[1. 1.]",
    "reasoning": "For state 0, the best action is to go to state 1 and get a reward of 1. For state 1, taking action 1 gives a reward of 1 and ends the episode, so its value is 1."
  },
  "category": "Reinforcement Learning",
  "starter_code": "import numpy as np\n\ndef bellman_update(V, transitions, gamma):\n    \"\"\"\n    Perform one step of value iteration using the Bellman equation.\n    Args:\n      V: np.ndarray, state values, shape (n_states,)\n      transitions: list of dicts. transitions[s][a] is a list of (prob, next_state, reward, done)\n      gamma: float, discount factor\n    Returns:\n      np.ndarray, updated state values\n    \"\"\"\n    # TODO: Implement Bellman update\n    pass",
  "learn_section": "# **The Bellman Equation**\n\nThe **Bellman equation** is a fundamental recursive equation in reinforcement learning that relates the value of a state to the values of possible next states. It provides the mathematical foundation for key RL algorithms such as value iteration and Q-learning.\n\n---\n\n## **Key Idea**\nFor each state $s$, the value $V(s)$ is the maximum expected return obtainable by choosing the best action $a$ and then following the optimal policy:\n\n$$\nV(s) = \\max_{a} \\sum_{s'} P(s'|s, a) \\left[ R(s, a, s') + \\gamma V(s') \\right]\n$$\n\nWhere:\n- $V(s)$: value of state $s$\n- $a$: possible actions\n- $P(s'|s, a)$: probability of moving to state $s'$ from $s$ via $a$\n- $R(s, a, s')$: reward for this transition\n- $\\gamma$: discount factor ($0 \\leq \\gamma \\leq 1$)\n- $V(s')$: value of next state\n\n---\n\n## **How to Use**\n1. **For each state:**\n   - For each possible action, sum over possible next states, weighting by transition probability.\n   - Add the immediate reward and the discounted value of the next state.\n   - Choose the action with the highest expected value (for control).\n2. **Repeat until values converge** (value iteration) or as part of other RL updates.\n\n---\n\n## **Applications**\n- **Value Iteration** and **Policy Iteration** in Markov Decision Processes (MDP)\n- **Q-learning** and other RL algorithms\n- Calculating the optimal value function and policy in gridworlds, games, and general MDPs\n\n---\n\n## **Why It Matters**\n- The Bellman equation formalizes the notion of **optimality** in sequential decision-making.\n- It is a backbone for teaching agents to solve environments with rewards, uncertainty, and long-term planning.",
  "contributor": [
    {
      "profile_link": "https://github.com/moe18",
      "name": "Moe Chabot"
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
