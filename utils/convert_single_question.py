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
    "id": "141",
    "description": "Write a Python function `convert_range` that shifts and scales the values of a NumPy array from their original range $[a, b]$ (where $a=\\min(x)$ and $b=\\max(x)$) to a new target range $[c, d]$. Your function should work for both 1D and 2D arrays, returning an array of the same shape, and only use NumPy. Return floating-point results, and ensure you use the correct formula to map the input interval to the output interval.",
    "test_cases": [
        {
            "test": "import numpy as np\nseq = np.array([388, 242, 124, 384, 313, 277, 339, 302, 268, 392])\nc, d = 0, 1\nout = convert_range(seq, c, d)\nprint(np.round(out, 6))",
            "expected_output": "[0.985075, 0.440299, 0.,       0.970149, 0.705224, 0.570896, 0.802239, 0.664179, 0.537313, 1.      ]"
        },
        {
            "test": "import numpy as np\nseq = np.array([[2028, 4522], [1412, 2502], [3414, 3694], [1747, 1233], [1862, 4868]])\nc, d = 4, 8\nout = convert_range(seq, c, d)\nprint(np.round(out, 6))",
            "expected_output": "[[4.874828 7.619257]\n [4.196974 5.396424]\n [6.4      6.708116]\n [4.565612 4.      ]\n [4.69216  8.      ]]"
        }
    ],
    "solution": "import numpy as np\n\ndef convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:\n    \"\"\"\n    Shift and scale values from their original range [min, max] to a target [c, d] range.\n\n    Parameters\n    ----------\n    values : np.ndarray\n        Input array (1D or 2D) to be rescaled.\n    c : float\n        New range lower bound.\n    d : float\n        New range upper bound.\n\n    Returns\n    -------\n    np.ndarray\n        Scaled array with the same shape as the input.\n    \"\"\"\n    a, b = values.min(), values.max()\n    return c + (d - c) / (b - a) * (values - a)",
    "example": {
        "input": "import numpy as np\nx = np.array([0, 5, 10])\nc, d = 2, 4\nprint(convert_range(x, c, d))",
        "output": "[2. 3. 4.]",
        "reasoning": "The minimum value (a) is 0 and the maximum value (b) is 10. The formula maps 0 to 2, 5 to 3, and 10 to 4 using: f(x) = c + (d-c)/(b-a)*(x-a)."
    },
    "category": "Machine Learning",
    "starter_code": "import numpy as np\n\ndef convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:\n    \"\"\"\n    Shift and scale values from their original range [min, max] to a target [c, d] range.\n    \"\"\"\n    # Your code here\n    pass",
    "title": "Shift and Scale Array to Target Range",
    "learn_section": "# **Shifting and Scaling a Range (Rescaling Data)**\n\n## **1. Motivation**\n\nRescaling (or shifting and scaling) is a common preprocessing step in data analysis and machine learning. It's often necessary to map data from an original range (e.g., test scores, pixel values, GPA) to a new range suitable for downstream tasks or compatibility between datasets. For example, you might want to shift a GPA from $[0, 10]$ to $[0, 4]$ for comparison or model input.\n\n---\n\n## **2. The General Mapping Formula**\n\nSuppose you have input values in the range $[a, b]$ and you want to map them to the interval $[c, d]$.\n\n- First, shift the lower bound to $0$ by applying $x \\mapsto x - a$, so $[a, b] \\rightarrow [0, b-a]$.\n- Next, scale to unit interval: $t \\mapsto \\frac{1}{b-a} \\cdot t$, yielding $[0, 1]$.\n- Now, scale to $[0, d-c]$ with $t \\mapsto (d-c)t$, and shift to $[c, d]$ with $t \\mapsto c + t$.\n- Combining all steps, the complete formula is:\n\n$$\n    f(x) = c + \\left(\\frac{d-c}{b-a}\\right)(x-a)\n$$\n\n- $x$ = the input value\n- $a = \\min(x)$ and $b = \\max(x)$\n- $c$, $d$ = target interval endpoints\n\n---\n\n## **3. Applications**\n- **Image Processing**: Rescale pixel intensities\n- **Feature Engineering**: Normalize features to a common range\n- **Score Conversion**: Convert test scores or grades between systems\n\n---\n\n## **4. Practical Considerations**\n- Be aware of the case when $a = b$ (constant input); this may require special handling (e.g., output all $c$).\n- For multidimensional arrays, use NumPy’s `.min()` and `.max()` to determine the full input range.\n\n---\n\nThis formula gives a **simple, mathematically justified way to shift and scale data to any target range**—a core tool for robust machine learning pipelines.\n",
    "contributor": [
        {
            "profile_link": "https://github.com/turkunov",
            "name": "turkunov"
        }
    ],
    "likes": "0",
    "dislikes": "0",
    "difficulty": "easy",
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
