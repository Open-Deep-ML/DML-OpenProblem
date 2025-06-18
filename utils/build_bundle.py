#!/usr/bin/env python
"""
Walk questions/* folders → emit build/<id>.json bundles.
"""

import json, pathlib, re

ROOT   = pathlib.Path("questions")
OUTDIR = pathlib.Path("build")
OUTDIR.mkdir(exist_ok=True)

def load_text(p):   return p.read_text(encoding="utf-8").rstrip("\n")
def load_json(p):   return json.loads(load_text(p))

def bundle_one(folder: pathlib.Path):
    meta = load_json(folder / "meta.json")

    meta["description"]   = load_text(folder / "description.md")
    meta["learn_section"] = load_text(folder / "learn.md")
    meta["starter_code"]  = load_text(folder / "starter_code.py")
    meta["solution"]      = load_text(folder / "solution.py")
    meta["example"]       = load_json(folder / "example.json")
    meta["test_cases"]    = load_json(folder / "tests.json")

    if "marimo_link" in meta:
        meta["marimo_link"] = meta["marimo_link"]


    for lang in ("tinygrad", "pytorch"):
        sub = folder / lang
        if sub.exists():
            if (sub / "starter_code.py").exists():
                meta[f"{lang}_starter_code"] = load_text(sub / "starter_code.py")
            if (sub / "solution.py").exists():
                meta[f"{lang}_solution"] = load_text(sub / "solution.py")
            if (sub / "tests.json").exists():
                meta[f"{lang}_test_cases"] = load_json(sub / "tests.json")

    out_path = OUTDIR / f"{meta['id']}.json"
    out_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"✓ bundled {out_path.name}")

def main():
    for qdir in sorted(p for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith('_')):
        bundle_one(qdir)

if __name__ == "__main__":
    main()
