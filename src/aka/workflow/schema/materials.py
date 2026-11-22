"""Fixed AKA materials and manifest-last publication of complete Schema designs.

Resources own input snapshots. This module owns only design files and their
commit manifest; the workflow's existing operation lock/root state owns intent
selection and subsequent DesignRef registration.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Mapping, Sequence

from aka.contracts.agent_resources import ResourcesSnapshot, WorkspaceResourceContribution
from aka.core.filesystem import (durable_write_json, durable_write_text,
                                 ensure_private_directory, fsync_directory)
from aka.resources.snapshots import inspect_snapshot

from .language import (LANGUAGE_VERSION, CheckPolicy, CheckedDesign,
                       check_design, collect_design, effective_yaml)
from .language.yaml_source import read_source_bytes

ASSETS = Path(__file__).parent / "assets"
DESIGN_FORMAT = "aka.schema.design.v1"


def _json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def closure_digest(files: Mapping[str, bytes]) -> str:
    """Byte-set identity; equal replacements can be handled as keep by the owner."""
    return _digest(_json({name: _digest(raw) for name, raw in sorted(files.items())}))


def _directory(path) -> Path:
    path = Path(path).absolute()
    if path.resolve() != path or any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Schema storage must not traverse links or aliases")
    return path


def _member(name: str) -> PurePosixPath:
    if not isinstance(name, str):
        raise ValueError("material name must be a string")
    path = PurePosixPath(name)
    if (not path.parts or path.is_absolute() or ".." in path.parts or path.as_posix() != name
            or "\\" in name or any(ord(c) < 32 for c in name)):
        raise ValueError(f"unsafe material name: {name!r}")
    return path


def select_materials(problem: str = "", *, wiki_ids: Sequence[str] | None = None,
                     max_wiki: int = 3) -> tuple[WorkspaceResourceContribution, ...]:
    """Choose fixed docs/starter and a bounded explicit or lexical Wiki selection.

    No dynamic provider, model, catalog SOTA inference or external repository
    lookup participates. Empty Wiki selection is a valid design starting point.
    """
    if type(max_wiki) is not int or max_wiki < 0:
        raise ValueError("max_wiki must be a nonnegative integer")
    from .release import verify_release
    verify_release()
    index = json.loads(read_source_bytes(ASSETS / "wiki-index.json"))
    entries = {row["id"]: row for row in index["entries"]}
    if wiki_ids is None:
        def words(text):
            # Common operator spelling variants, without treating dtype alone
            # as evidence that an unrelated implementation is useful.
            aliases = {"scaled": "scale", "grouped": "group", "matmul": "gemm",
                       "attention": "fmha"}
            return {aliases.get(word, word) for word in re.findall(r"[a-z0-9]+", text.lower())}
        query = words(problem)
        generic = {"fp8", "fp16", "fp32", "bf16", "int8", "kernel", "cuda", "triton"}
        scores = [(len(query & words(row["id"])) if (query & words(row["id"])) - generic else 0, name)
                  for name, row in entries.items()]
        selected = [name for score, name in sorted(scores, key=lambda x: (-x[0], x[1])) if score > 0][:max_wiki]
    else:
        selected = list(dict.fromkeys(wiki_ids))
        if len(selected) > max_wiki or any(name not in entries for name in selected):
            raise ValueError("explicit Wiki selection is unknown or exceeds max_wiki")
    resources = [WorkspaceResourceContribution("schema-specifications", ASSETS / "docs", "schema-materials/docs",
                                               version=LANGUAGE_VERSION),
                 WorkspaceResourceContribution("schema-starter", ASSETS / "ks/schema_template/starter.yaml",
                                               "schema-materials/starter.yaml", version=LANGUAGE_VERSION)]
    for index, name in enumerate(selected):
        row = entries[name]
        directory = ASSETS / "ks/schema_wiki" / _member(row["directory"])
        for number, filename in enumerate(row["files"]):
            _member(filename)
            resources.append(WorkspaceResourceContribution(f"schema-wiki-{index}-{number}", directory / filename,
                f"schema-materials/wiki/{name}/{filename}", version=LANGUAGE_VERSION))
    return tuple(resources)


def snapshot_inputs(destination, resources: Sequence[WorkspaceResourceContribution]) -> ResourcesSnapshot:
    """Freeze the explicitly public resource selection using the existing owner."""
    from aka.resources.plugins import PluginRegistry
    from aka.resources.wiki.providers import selected_provider
    if selected_provider() is not None:
        raise ValueError("Schema materials require an explicit selection without a dynamic Wiki provider")
    resources = tuple(resources)
    if any(item.visibility != "agent" for item in resources):
        raise ValueError("Schema role materials must be explicitly public Agent resources")
    registry = PluginRegistry(plugin_dir="", resources=resources, agent_skills=(), plugins=())
    return registry.export_snapshot(_directory(destination))


def material_view(snapshot: ResourcesSnapshot) -> dict[str, bytes]:
    """Read the exact role mount view from retained content, never original roots."""
    from aka.resources.snapshot_store import read_object
    facts = inspect_snapshot(snapshot)
    selection = facts.selection_lock
    if (selection["plugins"].get("plugins") or selection["agent_skills"]["skills"]):
        raise ValueError("Schema material snapshot must not supply executable plugins or skills")
    roots = {str(root.root): root for root in facts.roots}
    result = {}
    for resource in selection["workspace_resources"]["resources"]:
        if resource["visibility"] != "agent" or not resource["destination"]:
            raise ValueError("Schema material view cannot expose runtime-only inputs")
        root = roots[resource["source"]]
        destination = _member(resource["destination"])
        for entry in root.files:
            name = str(destination if root.kind == "file" else destination / _member(entry.path))
            if name in result:
                raise ValueError("Schema material destinations overlap")
            result[name] = read_object(snapshot, entry.sha256)
    return result


def role_material_view(snapshot: ResourcesSnapshot) -> dict[str, bytes]:
    """Expose only selected, byte-verified AKA specifications and Wiki references.

    Public Task text is supplied by the Task owner separately. In particular an
    arbitrary resource labelled 'agent' is not enough authority to expose a
    reference implementation, evaluator, launch configuration or credentials.
    """
    selected = material_view(snapshot)
    allowed = {"schema-materials/docs/" + path.name: path
               for path in (ASSETS / "docs").glob("*.md")}
    allowed["schema-materials/starter.yaml"] = ASSETS / "ks/schema_template/starter.yaml"
    index = json.loads(read_source_bytes(ASSETS / "wiki-index.json"))
    entries = {row["id"]: row for row in index["entries"]}
    for name, row in entries.items():
        for filename in row["files"]:
            allowed[f"schema-materials/wiki/{name}/{filename}"] = (
                ASSETS / "ks/schema_wiki" / row["directory"] / filename)
    view = {}
    wiki = set()
    for name, raw in selected.items():
        if not name.startswith("schema-materials/"):
            continue  # ResourcesSnapshot can also contain native Coder resources.
        if name not in allowed or read_source_bytes(allowed[name]) != raw:
            raise ValueError(f"unapproved or changed Schema role material: {name}")
        view[name] = raw
        if name.startswith("schema-materials/wiki/"):
            wiki.add(name.split("/")[2])
    if len(wiki) > 3:
        raise ValueError("Schema role view supports at most three selected Wiki references")
    for name in wiki:
        if any(f"schema-materials/wiki/{name}/{filename}" not in view for filename in entries[name]["files"]):
            raise ValueError("selected Wiki reference has an incomplete file closure: " + name)
    sources = json.loads(read_source_bytes(ASSETS / "SOURCES.json"))
    view["schema-materials/selection.json"] = _json({
        "snapshot_identity": snapshot.identity,
        "repository": sources["repository"], "commit": sources["commit"],
        "wiki": [{"id": name, "source_directory": entries[name]["directory"],
                  "status": "reference_only", "hardware_compatibility": "unproven"}
                 for name in sorted(wiki)],
        "files": {name: _digest(raw) for name, raw in sorted(view.items())},
        "guidance": "Wiki references are optional. Prove each hardware mechanism against the target; Blackwell-specific mechanisms do not establish L20D support.",
    })
    return view


def snapshot_ref(snapshot: ResourcesSnapshot) -> dict:
    inspect_snapshot(snapshot)
    return {"directory": str(snapshot.directory), "identity": snapshot.identity,
            "content_directory": str(snapshot.content_directory) if snapshot.content_directory else None}


def _snapshot_from_ref(value: Mapping) -> ResourcesSnapshot:
    return ResourcesSnapshot(Path(value["directory"]), value["identity"],
                             Path(value["content_directory"]) if value.get("content_directory") else None)


@dataclass(frozen=True)
class DesignRef:
    directory: Path
    identity: str

    def as_dict(self):
        return {"directory": str(self.directory), "identity": self.identity}

    @classmethod
    def from_dict(cls, value):
        return cls(Path(value["directory"]), value["identity"])


@dataclass(frozen=True)
class DesignArtifact:
    ref: DesignRef
    manifest: dict
    checked: CheckedDesign

    def effective_yaml(self, node="root") -> str:
        return effective_yaml(self.checked.resolved, node)


def _manifest(checked, *, version, parent, prepared_id, operation_id, mode,
              materials_snapshot, approval_refs):
    if type(version) is not int or version < 1:
        raise ValueError("design version must be a positive integer")
    if not all(isinstance(value, str) and value.strip() for value in (prepared_id, operation_id)):
        raise ValueError("prepared and operation identities must be nonempty")
    if mode not in {"design", "extract", "optimize", "import"}:
        raise ValueError("unknown design operation")
    if mode != "extract" and approval_refs:
        raise ValueError("design/optimize/import cannot inherit extraction approvals")
    if parent is not None:
        previous = read_design(parent)
        if previous.manifest["prepared_id"] != prepared_id or version <= previous.manifest["version"]:
            raise ValueError("new design must keep the prepared identity and advance its version")
    return {"format": DESIGN_FORMAT, "version": version,
            "parent": parent.as_dict() if parent else None,
            "prepared_id": prepared_id, "operation_id": operation_id, "mode": mode,
            "language_version": LANGUAGE_VERSION,
            "materials_snapshot": snapshot_ref(materials_snapshot),
            "files": {name: _digest(raw) for name, raw in sorted(checked.files.items())},
            "closure_digest": closure_digest(checked.files), "check_policy": checked.policy.as_dict(),
            "approval_refs": list(approval_refs),
            "origins": {name: checked.resolved.origins_dict(name) for name in checked.resolved.raw_task},
            "node_sources": {name: checked.resolved.source_manifest(name) for name in checked.resolved.raw_task}}


def publish_design(directory, source, *, version: int, prepared_id: str, operation_id: str,
                   materials_snapshot: ResourcesSnapshot, parent: DesignRef | None = None,
                   mode: str = "design", policy: CheckPolicy | None = None,
                   approval_refs: Sequence = ()) -> DesignRef:
    """Commit one complete immutable design; caller holds its operation write lock.

    Matching committed replay returns its original reference. An interrupted
    pre-manifest directory is preserved as a draft and must be rebuilt in a fresh
    host directory from the same retained candidate, without another model call.
    Root-state registration follows this function and can be retried from the
    committed manifest and durable intent without republishing or reauthoring.
    """
    directory = _directory(directory)
    checked = check_design(source, policy=policy)
    manifest = _manifest(checked, version=version, parent=parent, prepared_id=prepared_id,
        operation_id=operation_id, mode=mode, materials_snapshot=materials_snapshot, approval_refs=approval_refs)
    expected = _json(manifest)
    if (directory / "manifest.json").exists():
        existing = read_design(directory)
        if existing.ref.identity != _digest(expected):
            raise ValueError("committed design replay has conflicting operation or bytes")
        return existing.ref
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("uncommitted design directory retained; rebuild the same candidate in a fresh host directory")
    ensure_private_directory(directory)
    durable_write_json(directory / "intent.json", {"format": "aka.schema.design-intent.v1",
        "operation_id": operation_id, "manifest_sha256": _digest(expected),
        "closure_digest": manifest["closure_digest"]})
    raw_dir = directory / "raw"
    ensure_private_directory(raw_dir)
    for name, raw in sorted(checked.files.items()):
        durable_write_text(raw_dir / name, raw.decode("utf-8"), mode=0o444)
    if collect_design(raw_dir) != checked.files:
        raise ValueError("design bytes changed before manifest commit")
    # No externally visible commit exists until every data file has been fsynced.
    durable_write_text(directory / "manifest.json", expected.decode("utf-8"), mode=0o444)
    fsync_directory(directory)
    return read_design(DesignRef(directory, _digest(expected))).ref


def read_design(reference: DesignRef | str | Path) -> DesignArtifact:
    """Accept only a committed collection matching its manifest and frozen inputs."""
    directory = _directory(reference.directory if isinstance(reference, DesignRef) else reference)
    try:
        raw = read_source_bytes(directory / "manifest.json")
    except FileNotFoundError as error:
        raise ValueError("design has no committed manifest") from error
    identity = _digest(raw)
    if isinstance(reference, DesignRef) and identity != reference.identity:
        raise ValueError("design manifest identity changed")
    manifest = json.loads(raw)
    if manifest.get("format") != DESIGN_FORMAT or manifest.get("language_version") != LANGUAGE_VERSION:
        raise ValueError("unsupported design format or language identity")
    if {p.name for p in directory.iterdir()} != {"intent.json", "raw", "manifest.json"}:
        raise ValueError("published design has unexpected or missing files")
    intent = json.loads(read_source_bytes(directory / "intent.json"))
    if (intent.get("manifest_sha256") != identity or intent.get("operation_id") != manifest["operation_id"]
            or intent.get("closure_digest") != manifest["closure_digest"]):
        raise ValueError("published design does not match its durable intent")
    files = collect_design(directory / "raw")
    if ({name: _digest(value) for name, value in files.items()} != manifest["files"]
            or closure_digest(files) != manifest["closure_digest"]):
        raise ValueError("published design bytes differ from the committed closure")
    inspect_snapshot(_snapshot_from_ref(manifest["materials_snapshot"]))
    checked = check_design(files, policy=CheckPolicy(**manifest["check_policy"]))
    if (manifest["origins"] != {name: checked.resolved.origins_dict(name) for name in checked.resolved.raw_task}
            or manifest["node_sources"] != {name: checked.resolved.source_manifest(name) for name in checked.resolved.raw_task}):
        raise ValueError("published design provenance differs from its actual providers")
    return DesignArtifact(DesignRef(directory, identity), manifest, checked)
