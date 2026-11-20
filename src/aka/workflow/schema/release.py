"""Explicit source release selection. Called only after Schema is enabled."""
from __future__ import annotations

from importlib import metadata
from importlib.resources import files
from importlib.util import find_spec
import hashlib
import json
from pathlib import Path
import stat


class SchemaUnavailable(ValueError):
    pass


def verify_release() -> dict:
    """Verify the installed KS code/resources against the shipped selected manifest.

    No checkout search, git command, network call or automatic installation occurs.
    An exact extracted checkout is supported as well as the detached submodule.
    """
    raw = files(__package__).joinpath("release.json").read_bytes()
    release = json.loads(raw)
    spec = find_spec("kernelschema")
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        raise SchemaUnavailable(
            "Schema needs the fixed KernelSchema source/editable installation; "
            "see docs/schema-optional-module.md")
    root = Path(spec.origin).resolve().parent.parent
    for name, expected in release["files"].items():
        path = root / name
        try:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or path.resolve() != path:
                raise OSError("not a regular selected release file")
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise SchemaUnavailable(f"fixed KernelSchema resource missing: {name}") from error
        if observed != expected["sha256"] or stat.S_IMODE(info.st_mode) != expected["mode"]:
            raise SchemaUnavailable(f"fixed KernelSchema resource changed: {name}")
    # No unverified KS initializer/backend runs while selecting this release.
    import kernelschema
    from kernelschema.resources import checkout_root
    if (Path(kernelschema.__file__).resolve().parent != root / "kernelschema"
            or checkout_root().resolve() != root):
        raise SchemaUnavailable("KernelSchema import and resource roots differ")
    try:
        installed = metadata.version("atrex-aka-schema")
    except metadata.PackageNotFoundError as error:
        raise SchemaUnavailable("install the matching atrex-aka-schema package") from error
    return {"aka_distribution": "atrex-aka-schema", "aka_version": installed,
            "ks_commit": release["commit"], "ks_url": release["url"],
            "ks_root": str(root), "ks_module": str(Path(kernelschema.__file__).resolve()),
            "manifest_sha256": "sha256:" + hashlib.sha256(raw).hexdigest()}
