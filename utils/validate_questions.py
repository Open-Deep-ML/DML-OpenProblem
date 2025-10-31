#!/usr/bin/env python
"""
Validate build/*.json (or files passed on the CLI) against schemas/question.schema.json
"""

import json
import pathlib
import sys
from json.decoder import JSONDecodeError
from jsonschema import Draft7Validator, exceptions as js_exceptions

HERE = pathlib.Path(__file__).resolve().parent
SCHEMA_PATH = (HERE / ".." / "schemas" / "question.schema.json").resolve()

def load_json_strict(path: pathlib.Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except JSONDecodeError as e:
        # Helpful context around the error location
        text = path.read_text(encoding="utf-8", errors="replace")
        start = max(e.pos - 60, 0)
        end = min(e.pos + 60, len(text))
        snippet = text[start:end]
        pointer = " " * (e.pos - start) + "^"
        print(f"\n❌ JSON parse error in {path} @ line {e.lineno}, col {e.colno}: {e.msg}")
        print("   Context:")
        print(snippet)
        print(pointer)
        raise

def load_schema():
    if not SCHEMA_PATH.exists():
        print(f"❌ Schema not found at {SCHEMA_PATH}")
        sys.exit(1)
    schema = load_json_strict(SCHEMA_PATH)
    try:
        Draft7Validator.check_schema(schema)
    except js_exceptions.SchemaError as se:
        print("❌ The schema file is not a valid Draft-07 JSON Schema.")
        print("   ", se.message)
        sys.exit(1)
    return schema

def validate_file(validator: Draft7Validator, fp: pathlib.Path) -> bool:
    try:
        data = load_json_strict(fp)
    except JSONDecodeError:
        print(f"   (Could not parse {fp.name} as JSON.)")
        return False

    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    if errors:
        print(f"❌ {fp.name}")
        for e in errors:
            path = "/".join(map(str, e.path)) or "(root)"
            print("   •", path, "-", e.message)
        return False
    else:
        print(f"✓ {fp.name}")
        return True

def main():
    schema = load_schema()
    validator = Draft7Validator(schema)

    # Files passed on CLI? Use those. Otherwise default to build/*.json
    args = [pathlib.Path(a) for a in sys.argv[1:]]
    if args:
        files = []
        for a in args:
            if a.is_dir():
                files.extend(sorted(a.glob("*.json")))
            else:
                files.append(a)
    else:
        files = sorted((HERE / ".." / "build").resolve().glob("*.json"))

    if not files:
        print("No JSON files found to validate.")
        sys.exit(0)

    ok = True
    for fp in files:
        ok = validate_file(validator, fp) and ok

    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
