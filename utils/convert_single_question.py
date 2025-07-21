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
    "id": "164",
    "title": "Gambler's Problem: Value Iteration",
    "description": "A gambler has the chance to bet on a sequence of coin flips. If the coin lands heads, the gambler wins the amount staked; if tails, the gambler loses the stake. The goal is to reach 100, starting from a given capital $s$ (with $0 < s < 100$). The game ends when the gambler reaches $0$ (bankruptcy) or $100$ (goal). On each flip, the gambler can bet any integer amount from $1$ up to $\\min(s, 100-s)$.\n\nThe probability of heads is $p_h$ (known). Reward is $+1$ if the gambler reaches $100$ in a transition, $0$ otherwise.\n\n**Your Task:**\nWrite a function `gambler_value_iteration(ph, theta=1e-9)` that:\n- Computes the optimal state-value function $V(s)$ for all $s = 1, ..., 99$ using value iteration.\n- Returns the optimal policy as a mapping from state $s$ to the optimal stake $a^*$ (can return any optimal stake if there are ties).\n\n**Inputs:**\n- `ph`: probability of heads (float between 0 and 1)\n- `theta`: threshold for value iteration convergence (default $1e-9$)\n\n**Returns:**\n- `V`: array/list of length 101, $V[s]$ is the value for state $s$\n- `policy`: array/list of length 101, $policy[s]$ is the optimal stake in state $s$ (0 if $s=0$ or $s=100$)\n",
    "test_cases": [
      {
        "test": "ph = 0.4\nV, policy = gambler_value_iteration(ph)\nprint(round(V[50], 4))\nprint(policy[50])",
        "expected_output": "0.4\n50"
      },
      {
        "test": "ph = 0.25\nV, policy = gambler_value_iteration(ph)\nprint(round(V[80], 4))\nprint(policy[80])",
        "expected_output": "0.4534\n5"
      }
    ],
    "solution": """def gambler_value_iteration(ph, theta=1e-9):
    # Initialize value function for states 0 to 100; terminal states 0 and 100 have value 0
    V = [0.0] * 101
    # Initialize policy array (bet amount for each state)
    policy = [0] * 101
    
    # Value iteration loop
    while True:
        delta = 0
        # Iterate over non-terminal states (1 to 99)
        for s in range(1, 100):
            # Possible actions: bet between 1 and min(s, 100 - s)
            actions = range(1, min(s, 100 - s) + 1)
            action_returns = []
            # Evaluate each action
            for a in actions:
                win_state = s + a
                lose_state = s - a
                # Reward is 1 if transition reaches 100, else 0
                reward = 1.0 if win_state == 100 else 0.0
                # Expected value: ph * (reward + V[win]) + (1 - ph) * V[lose]
                ret = ph * (reward + V[win_state]) + (1 - ph) * V[lose_state]
                action_returns.append(ret)
            # Update V[s] with the maximum expected value
            max_value = max(action_returns)
            delta = max(delta, abs(V[s] - max_value))
            V[s] = max_value
        # Check for convergence
        if delta < theta:
            break
    
    # Extract optimal policy
    for s in range(1, 100):
        actions = range(1, min(s, 100 - s) + 1)
        best_action = 0
        best_return = -float('inf')
        # Find action that maximizes expected value
        for a in actions:
            win_state = s + a
            lose_state = s - a
            reward = 1.0 if win_state == 100 else 0.0
            ret = ph * (reward + V[win_state]) + (1 - ph) * V[lose_state]
            if ret > best_return:
                best_return = ret
                best_action = a
        policy[s] = best_action
    
    return V, policy""",
    "example": {
      "input": "ph = 0.4\nV, policy = gambler_value_iteration(ph)\nprint(round(V[50], 4))\nprint(policy[50])",
      "output": "0.0178\n1",
      "reasoning": "From state 50, the optimal action is to bet 1, with a probability of reaching 100 of about 0.0178 when ph=0.4."
    },
    "category": "Reinforcement Learning",
    "starter_code": "def gambler_value_iteration(ph, theta=1e-9):\n    \"\"\"\n    Computes the optimal value function and policy for the Gambler's Problem.\n    Args:\n      ph: probability of heads\n      theta: convergence threshold\n    Returns:\n      V: list of values for all states 0..100\n      policy: list of optimal stakes for all states 0..100\n    \"\"\"\n    # Your code here\n    pass",
    "learn_section": "# **Gambler's Problem and Value Iteration**\n\nIn the Gambler's Problem, a gambler repeatedly bets on a coin flip with probability $p_h$ of heads. The goal is to reach 100 starting from some capital $s$. At each state, the gambler chooses a stake $a$ (between $1$ and $\\min(s, 100-s)$). If heads, the gambler gains $a$; if tails, loses $a$. The game ends at $0$ or $100$.\n\nThe objective is to find the policy that maximizes the probability of reaching 100 (the state-value function $V(s)$ gives this probability). The value iteration update is:\n\n$$\nV(s) = \\max_{a \\in \\text{Actions}(s)} \\Big[ p_h (\\text{reward} + V(s + a)) + (1-p_h)V(s-a) \\Big]\n$$\n\nwhere the reward is $+1$ only if $s + a = 100$.\n\nAfter convergence, the greedy policy chooses the stake maximizing this value. This is a classic episodic MDP, and the optimal policy may not be unique (ties are possible).",
    "contributor": [
      {
        "profile_link": "https://github.com/moe18",
        "name": "Moe Chabot"
      }
    ],
    "likes": "0",
    "dislikes": "0",
    "difficulty": "hard",
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
