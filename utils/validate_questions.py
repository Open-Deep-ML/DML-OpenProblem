#!/usr/bin/env python
"""
Validate build/*.json against schemas/question.schema.json
"""

import json
import pathlib
import sys
from jsonschema import Draft7Validator

SCHEMA = json.load(open("schemas/question.schema.json"))


def validate_file(fp: pathlib.Path):
    data = json.load(fp.open())
    errors = list(Draft7Validator(SCHEMA).iter_errors(data))
    if errors:
        print(f"❌ {fp.name}")
        for e in errors:
            path = "/".join(map(str, e.path))
            print("   •", path, e.message)
        return False
    print(f"✓ {fp.name}")
    return True


def main():
    build_dir = pathlib.Path("build")
    ok = all(validate_file(f) for f in build_dir.glob("*.json"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
