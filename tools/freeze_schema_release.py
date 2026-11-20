#!/usr/bin/env python3
"""Freeze selected fixed submodule code/resources; never fetch or initialize it."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SELECTED = ("kernelschema", "schema_template", "schema_extract", "schema_docs",
            "examples/workflows/row_softmax", "tests/workflows/fixtures/v1/row_softmax",
            "pyproject.toml", "README.md", "DESIGN_SCHEMA.md", "OPTIMIZE_SCHEMA.md",
            "IMPLEMENT_SCHEMA.md", "EXTRACT_SCHEMA.md")


def freeze(commit):
    checkout = ROOT / "optional/KernelSchema"
    def git(*args):
        return subprocess.check_output(["git", "-C", str(checkout), *args])
    if git("rev-parse", "HEAD").decode().strip() != commit:
        raise ValueError("checkout must be detached at the selected full commit")
    if subprocess.run(["git", "-C", str(checkout), "symbolic-ref", "-q", "HEAD"],
                      stdout=subprocess.DEVNULL).returncode == 0:
        raise ValueError("selected checkout must have detached HEAD")
    subprocess.run(["git", "-C", str(checkout), "diff", "--quiet", commit, "--", *SELECTED], check=True)
    inventory = {}
    for row in git("ls-tree", "-r", "-z", commit, "--", *SELECTED).split(b"\0"):
        if not row:
            continue
        head, name = row.split(b"\t", 1)
        mode, kind, _ = head.split()
        name = name.decode()
        if name.startswith("kernelschema/runtime/qualification/"):
            continue  # Never publish private qualification captures.
        if kind != b"blob" or mode not in (b"100644", b"100755"):
            raise ValueError("release resources require regular files: " + name)
        inventory[name] = {"sha256": hashlib.sha256((checkout / name).read_bytes()).hexdigest(),
                           "mode": int(mode[-3:], 8)}
    release = {"format": "aka.schema.source-release.v1", "commit": commit,
        "url": "https://github.com/bcacdwk/KernelSchema.git",
        "aka_base_commit": "c984d141145d31795514579db213413a1bbb0775",
        "files": inventory}
    path = ROOT / "src/aka/workflow/schema/release.json"
    path.write_text(json.dumps(release, sort_keys=True, indent=2) + "\n")
    print(f"{len(inventory)} fixed selected files: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    freeze(parser.parse_args().commit)
