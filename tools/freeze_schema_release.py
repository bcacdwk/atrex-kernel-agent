#!/usr/bin/env python3
"""Freeze AKA Schema language/materials/prompts, independently of workflow code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "src/aka/workflow/schema"
MATERIAL_FILES = ("NOTICE.md",)
MATERIAL_DIRECTORIES = ("language", "assets", "prompts")


def inventory():
    selected = [SCHEMA / name for name in MATERIAL_FILES if (SCHEMA / name).exists()]
    for directory in MATERIAL_DIRECTORIES:
        selected.extend(path for path in (SCHEMA / directory).rglob("*")
                        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"})
    files = {}
    for path in sorted(selected):
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or path.resolve() != path:
            raise ValueError("release requires unaliased regular local files: " + str(path))
        raw = path.read_bytes()
        files[path.relative_to(SCHEMA).as_posix()] = {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
    provenance = json.loads((SCHEMA / "assets/SOURCES.json").read_text())
    return {"format": "aka.schema.release.v2", "language_version": "aka.schema.language.v1",
            "provenance": {"url": provenance["repository"], "commit": provenance["commit"]}, "files": files}


def freeze(*, check=False):
    path = SCHEMA / "release.json"
    value = json.dumps(inventory(), sort_keys=True, indent=2) + "\n"
    if check:
        if not path.exists() or path.read_text() != value:
            raise ValueError("AKA Schema local release is stale; run tools/freeze_schema_release.py")
    else:
        path.write_text(value)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the local inventory without changing it")
    args = parser.parse_args()
    try:
        result = freeze(check=args.check)
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + "\n")
    print(result)
