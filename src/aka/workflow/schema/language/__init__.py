"""AKA's Schema language: exact YAML, static checks and whole-region resolution.

The checker consumes one complete byte snapshot. It never discovers a checkout,
executes expressions, resolves Git references, or writes source files.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from aka.core.filesystem import file_signature

from . import rules, text_policy, yaml_source
from .design_resolution import (MAX_DESIGNS, RegionOrigin, ResolvedTask, effective_yaml,
                                read_directory, resolve_documents, task_files)
from .diagnostics import Diagnostic, collect_style_diagnostics, format_diagnostics
from .errors import CapacityError, SchemaError, ToolError, bounded
from .normalization import (LoadedDocument, NormalizedTask, load_for_normalization,
                            loads_for_normalization, normalize_documents)
from .rules import dispatch, values
from .yaml_source import Document, load, loads

LANGUAGE_VERSION = "aka.schema.language.v1"


@dataclass(frozen=True)
class CheckPolicy:
    """Style is advisory for design/optimize; extraction can request strict checks."""
    skip_hardware: bool = False
    style: str = "advisory"

    def __post_init__(self):
        if type(self.skip_hardware) is not bool or self.style not in {"strict", "advisory"}:
            raise ValueError("Schema policy requires a boolean skip_hardware and strict/advisory style")

    def as_dict(self):
        return {"skip_hardware": self.skip_hardware, "style": self.style}


@dataclass(frozen=True)
class CheckedDesign:
    files: dict[str, bytes]
    resolved: ResolvedTask
    diagnostics: tuple[Diagnostic, ...]
    policy: CheckPolicy
    language_version: str = LANGUAGE_VERSION


@bounded
def collect_design(directory: str | Path) -> dict[str, bytes]:
    """Freeze a flat directory and reject partial, extra, aliased or changing inputs."""
    directory = Path(directory).absolute()
    files = task_files(directory)
    entries = {path.name: path for path in files.values()}
    entries["dispatch.md"] = directory / "dispatch.md"
    try:
        signatures = {name: file_signature(path.lstat()) for name, path in entries.items()}
        result = {name: yaml_source.read_source_bytes(path) for name, path in entries.items()}
        if set(path.name for path in directory.iterdir()) != set(entries):
            raise SchemaError(f"{directory}: design inventory changed while reading")
        if any(file_signature(path.lstat()) != signatures[name] for name, path in entries.items()):
            raise SchemaError(f"{directory}: design files changed while reading")
    except FileNotFoundError as error:
        raise SchemaError(f"{directory}: complete design requires dispatch.md and root.yaml") from error
    except (OSError, yaml_source.ReadError) as error:
        raise SchemaError(str(error)) from error
    return result


def _snapshot(source: str | Path | Mapping[str, bytes | str]) -> dict[str, bytes]:
    if not isinstance(source, Mapping):
        return collect_design(source)
    result = {}
    for name, raw in source.items():
        if not isinstance(name, str) or Path(name).name != name or "\\" in name:
            raise SchemaError("design names must be top-level file names")
        if name != "dispatch.md":
            if not name.endswith(".yaml"):
                raise SchemaError(f"{name}: only <node>.yaml and dispatch.md belong in a design")
            errors = []
            values.node_id(name[:-5], name, errors)
            hard = [item for item in errors if not item.category.startswith("style.")]
            if hard:
                raise SchemaError(str(hard[0]))
        if isinstance(raw, str):
            try:
                raw = raw.encode("utf-8")
            except UnicodeError as error:
                raise SchemaError(f"{name}: expected UTF-8 text") from error
        if not isinstance(raw, bytes):
            raise SchemaError(f"{name}: expected immutable bytes or text")
        if len(raw) > yaml_source.MAX_FILE_BYTES:
            raise CapacityError(f"{name}: input exceeds {yaml_source.MAX_FILE_BYTES} bytes")
        result[name] = raw
    if "root.yaml" not in result or "dispatch.md" not in result:
        raise SchemaError("complete design requires root.yaml and dispatch.md")
    if len(result) - 1 > MAX_DESIGNS:
        raise CapacityError(f"at most {MAX_DESIGNS} YAML designs are supported")
    return result


def _documents(snapshot):
    files, documents = {}, {}
    for filename, raw in snapshot.items():
        if filename == "dispatch.md":
            continue
        name = filename[:-5]
        files[name] = Path(filename)
        try:
            documents[name] = loads(raw.decode("utf-8"), name, path=filename)
        except (UnicodeError, yaml_source.ReadError) as error:
            raise SchemaError(f"{filename}: {error}", diagnostics=(
                Diagnostic(name, str(error), category="syntax.yaml"),)) from error
    return files, documents


def _diagnostics(items, policy):
    result = []
    for item in items:
        if policy.style == "advisory" and item.category.startswith("style."):
            item = replace(item, severity="warning")
        if item not in result:
            result.append(item)
    return result


def _raise_diagnostics(items, files, documents, *, resolved=None):
    if any(item.severity == "error" for item in items):
        raise SchemaError(format_diagnostics(items, files, documents, resolved=resolved), diagnostics=items)


@bounded
def validate_documents(files: dict, documents: dict, *, policy: CheckPolicy | None = None,
                       notices: list[str] | None = None, diagnostics: list | None = None,
                       source_hashes: dict | None = None, skip_hardware=False) -> ResolvedTask:
    """Check parsed source once; no filesystem or external locator reads occur.

    This lower level entry defaults to strict style for extraction. A successful
    result proves supported structure/relations, not arbitrary Contract semantics.
    """
    policy = policy or CheckPolicy(skip_hardware=skip_hardware, style="strict")
    if files.keys() != documents.keys():
        raise ToolError("source files and parsed documents must name the same snapshot nodes")
    task = {name: document.value for name, document in documents.items()}
    with collect_style_diagnostics() as pipe_style:
        structure = rules.validate_structure(task, skip_hardware=policy.skip_hardware, allow_from=True)
        # Style cannot make a safe structure unsafe to traverse.
        _raise_diagnostics([d for d in structure if not d.category.startswith("style.")], files, documents)
        textual = [item for document in documents.values()
                   for item in text_policy.check_document(document, skip_hardware=policy.skip_hardware)]
        resolved = resolve_documents(files, documents)
        relations = rules.validate_relations(resolved.effective_task, skip_hardware=policy.skip_hardware)
    all_items = _diagnostics([*structure, *textual, *relations, *pipe_style], policy)
    if policy.skip_hardware:
        all_items.append(Diagnostic("root.CONTRACT.Hardware_specification",
            "Hardware structure and hardware-dependent relations were explicitly skipped",
            severity="warning", category="hardware.skipped"))
    if diagnostics is not None:
        diagnostics.extend(all_items)
    if notices is not None:
        notices.extend(str(item) for item in all_items if item.severity == "warning")
    _raise_diagnostics(all_items, files, documents, resolved=resolved)
    if source_hashes is not None:
        source_hashes.update(resolved.source_hashes)
    return resolved


def validate_dispatch(content: str, files: dict[str, Path]) -> None:
    errors = dispatch.validate(content, files)
    if errors:
        raise SchemaError("\n".join(map(str, errors)), diagnostics=errors)


@bounded
def check_design(source: str | Path | Mapping[str, bytes | str], *,
                 policy: CheckPolicy | None = None, skip_hardware: bool = False,
                 style: str = "advisory") -> CheckedDesign:
    """Validate every byte/file in a complete design, including dispatch closure."""
    policy = policy or CheckPolicy(skip_hardware=skip_hardware, style=style)
    snapshot = _snapshot(source)
    files, documents = _documents(snapshot)
    diagnostics = []
    resolved = validate_documents(files, documents, policy=policy, diagnostics=diagnostics)
    try:
        content = snapshot["dispatch.md"].decode("utf-8")
    except UnicodeError as error:
        raise SchemaError("dispatch.md: expected UTF-8 Markdown") from error
    validate_dispatch(content, files)
    return CheckedDesign(snapshot, resolved, tuple(diagnostics), policy)


@bounded
def normalize_design(source: str | Path | Mapping[str, bytes | str], *, skip_hardware=False) -> NormalizedTask:
    """Plan the entire group's edits without changing raw bytes or publishing them."""
    snapshot = _snapshot(source)
    files, documents = {}, {}
    for filename, raw in snapshot.items():
        if filename == "dispatch.md":
            continue
        name = filename[:-5]
        files[name] = Path(filename)
        try:
            documents[name] = loads_for_normalization(raw.decode("utf-8"), name,
                                                     path=filename, skip_hardware=skip_hardware)
        except (UnicodeError, yaml_source.ReadError) as error:
            raise SchemaError(f"{filename}: {error}") from error
    return normalize_documents(files, documents, skip_hardware=skip_hardware)


def json_material(value):
    """Expose exact decimals to roles without silently converting them to float."""
    from decimal import Decimal
    if isinstance(value, Decimal):
        return {"$decimal": str(value)}
    if isinstance(value, dict):
        return {key: json_material(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_material(child) for child in value]
    return value
