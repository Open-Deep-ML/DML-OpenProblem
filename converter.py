#!/usr/bin/env python
"""
utils/convert_all_problems.py
─────────────────────────────
Reads the monolithic questions dump and converts every entry into the
folder-based layout.  Optional fields (TinyGrad, PyTorch, marimo_link)
are copied only when they exist.

Run from repo root:
    python utils/convert_all_problems.py
"""

import base64, json, pathlib, re

SRC_JSON = pathlib.Path(
    "all_problems_06_15_2025.json"
)
OUT_ROOT = pathlib.Path("questions")
OUT_ROOT.mkdir(exist_ok=True)

# ───────── helpers ──────────────────────────────────────────────────────────
def slugify(txt: str) -> str:
    txt = re.sub(r"[^0-9A-Za-z]+", "-", txt.lower())
    return re.sub(r"-{2,}", "-", txt).strip("-")[:50]

def maybe_b64(s: str) -> str:
    try:
        if len(s) % 4 == 0 and re.fullmatch(r"[0-9A-Za-z+/=\n\r]+", s):
            return base64.b64decode(s).decode("utf-8")
    except Exception:
        pass
    return s

def w_text(path: pathlib.Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip("\n") + "\n", encoding="utf-8")

def w_json(path: pathlib.Path, obj):
    w_text(path, json.dumps(obj, indent=2, ensure_ascii=False))

# ───────── main conversion ─────────────────────────────────────────────────
def convert(q: dict):
    folder = OUT_ROOT / f"{q['id']}_{slugify(q['title'])}"
    folder.mkdir(parents=True, exist_ok=True)

    # --- meta.json ---------------------------------------------------------
    meta = {
        "id":          q["id"],
        "title":       q["title"],
        "difficulty":  q["difficulty"],
        "category":    q["category"],
        "video":       q.get("video", ""),
        "likes":       q.get("likes", "0"),
        "dislikes":    q.get("dislikes", "0"),
        "contributor": q.get("contributor", []),
        "tinygrad_difficulty": q.get("tinygrad_difficulty"),
        "pytorch_difficulty":  q.get("pytorch_difficulty")
    }
    # only include marimo_link when present
    if "marimo_link" in q:
        meta["marimo_link"] = q["marimo_link"]

    w_json(folder / "meta.json", meta)

    # --- core files --------------------------------------------------------
    w_text(folder / "description.md", q["description"])
    w_text(folder / "learn.md",       q["learn_section"])
    w_text(folder / "starter_code.py", q["starter_code"])
    w_text(folder / "solution.py",     q["solution"])
    w_json(folder / "example.json",    q["example"])
    w_json(folder / "tests.json",      q["test_cases"])

    # --- language-specific extras -----------------------------------------
    for lang in ("tinygrad", "pytorch"):
        sc, so, tc = (f"{lang}_starter_code", f"{lang}_solution", f"{lang}_test_cases")
        if any(k in q for k in (sc, so, tc)):
            sub = folder / lang
            if sc in q:
                w_text(sub / "starter_code.py", maybe_b64(q[sc]))
            if so in q:
                w_text(sub / "solution.py",     maybe_b64(q[so]))
            if tc in q:
                w_json(sub / "tests.json",      q[tc])

    print(f"✓ {folder.name}")

# ───────── driver ──────────────────────────────────────────────────────────
def main():
    data = json.loads(SRC_JSON.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list of questions.")
    for q in data:
        convert(q)

if __name__ == "__main__":
    if not SRC_JSON.exists():
        raise FileNotFoundError(SRC_JSON)
    main()
