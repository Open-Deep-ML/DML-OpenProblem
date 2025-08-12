#!/usr/bin/env python
import json
import pathlib

TEMPLATE = pathlib.Path("questions/_template")
LANGS = ["tinygrad", "pytorch"]


def w(path, txt):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(txt.rstrip() + "\n")


def main():
    META = {
        "id": "XXX",
        "title": "TITLE GOES HERE",
        "difficulty": "medium",
        "category": "Machine Learning",
        "video": "",
        "likes": "0",
        "dislikes": "0",
        "contributor": [],
        "tinygrad_difficulty": "",
        "pytorch_difficulty": "",
    }
    w(TEMPLATE / "meta.json", json.dumps(META, indent=2))
    w(TEMPLATE / "description.md", "## Problem\n\nDescribe the task.")
    w(TEMPLATE / "learn.md", "## Solution Explanation\n\nExplain here.")
    w(TEMPLATE / "starter_code.py", "def your_function(...):\n    pass")
    w(TEMPLATE / "solution.py", "def your_function(...):\n    ...")
    w(
        TEMPLATE / "example.json",
        json.dumps({"input": "...", "output": "...", "reasoning": "..."}, indent=2),
    )
    w(
        TEMPLATE / "tests.json",
        json.dumps(
            [{"test": "print(your_function(...))", "expected_output": "..."}], indent=2
        ),
    )
    for lang in LANGS:
        sub = TEMPLATE / lang
        w(sub / "starter_code.py", "def your_function(...):\n    pass")
        w(sub / "solution.py", "def your_function(...):\n    ...")
        w(
            sub / "tests.json",
            json.dumps(
                [{"test": "print(your_function(...))", "expected_output": "..."}],
                indent=2,
            ),
        )
    print("Template ready at questions/_template/")


if __name__ == "__main__":
    main()
