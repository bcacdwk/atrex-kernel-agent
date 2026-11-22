"""Verify AKA Schema language/materials/prompts without discovering upstream code."""
from __future__ import annotations

from importlib import metadata
import hashlib
import json
from pathlib import Path, PurePosixPath

from .language import LANGUAGE_VERSION, CapacityError
from .language.yaml_source import ReadError, read_source_bytes

ROOT = Path(__file__).parent


class SchemaUnavailable(ValueError):
    pass


def verify_release() -> dict:
    """Check language and role materials against the shipped local inventory.

    This works from a source checkout or unpacked wheel, with no upstream
    package, source checkout, Git object, command, parser or network access.
    """
    try:
        raw = read_source_bytes(ROOT / "release.json")
        release = json.loads(raw)
        if release.get("format") != "aka.schema.release.v2" or release.get("language_version") != LANGUAGE_VERSION:
            raise ValueError("unsupported AKA Schema release identity")
        for name, expected in release["files"].items():
            member = PurePosixPath(name)
            if (member.is_absolute() or ".." in member.parts or str(member) != name or "\\" in name
                    or (name != "NOTICE.md" and (len(member.parts) < 2 or member.parts[0] not in {"language", "assets", "prompts"}))):
                raise ValueError(f"invalid release member {name!r}")
            value = read_source_bytes(ROOT / name)
            if len(value) != expected["size"] or hashlib.sha256(value).hexdigest() != expected["sha256"]:
                raise ValueError(f"AKA Schema release resource changed: {name}")
    except (OSError, ValueError, KeyError, ReadError, CapacityError) as error:
        raise SchemaUnavailable(str(error)) from error
    try:
        installed = metadata.version("atrex-aka-schema")
    except metadata.PackageNotFoundError:
        installed = "source"
    return {"aka_distribution": "atrex-aka-schema", "aka_version": installed,
            "language_version": release["language_version"], "resource_root": str(ROOT.absolute()),
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "provenance": release["provenance"]}
