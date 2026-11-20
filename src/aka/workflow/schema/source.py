"""Exact single-file Source/native identities; no seed installation policy."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from aka.contracts.workspace import RevisionContent
from aka.task.source.formats.registry import resolve
from kernelschema.artifacts import Bundle, Publication, read_regular, source_manifest
from kernelschema.contracts import ProtocolError, digest, sha256


@dataclass(frozen=True)
class SourceBinding:
    source: dict
    native_identity: str
    kernel_sha256: str
    size: int
    mode: str = "0644"

    def as_dict(self):
        return dict(source=self.source, native_identity=self.native_identity,
                    kernel_sha256=self.kernel_sha256, size=self.size, mode=self.mode)


def selected_seed(bundle: Bundle, task: dict, seed: dict | None) -> dict:
    """null selects the Task starter; invalid explicit seeds never fall back."""
    selected = seed if seed is not None else bundle.document(task, "task")["starter_source"]
    if selected is None:
        raise ProtocolError("AKA row-softmax does not support empty_start")
    bundle.source(selected)
    return selected


def bind_source(bundle: Bundle, reference: dict, candidate_workspace) -> SourceBinding:
    data = bundle.source(reference)
    identity = candidate_workspace.source_identity(RevisionContent({"kernel.py": data}))
    return SourceBinding(reference, identity, sha256(data), len(data))


def freeze_native_source(publication: Publication, workspace: Path, candidate_workspace,
                         *, revision: str | None = None, expected: dict | None = None) -> SourceBinding:
    """Snapshot the actual native source, checking bytes AND original mode.

    solution.json is metadata only. Auxiliary declared/local imported source and
    non-single-file selections fail before any evaluator submission.
    """
    workspace = Path(workspace).absolute()
    if revision is not None:
        from aka.task.candidate_workspace.history import resolve_revision
        revision = resolve_revision(workspace, revision)
    system, policy = resolve(workspace, revision=revision)
    if system.name != "single_file" or policy is not None:
        raise ProtocolError("row-softmax Source requires native single_file")
    source = candidate_workspace.read_source(workspace, revision=revision)
    if set(source.files) != {"kernel.py"} or source.files["kernel.py"] is None:
        raise ProtocolError("complete source requires exactly kernel.py")
    data = source.files["kernel.py"]
    if revision is None:
        actual, mode = read_regular(workspace, "kernel.py")
        if actual != data:
            raise ProtocolError("native source changed during snapshot")
        limitation = system.typed_limitation(workspace, policy)
        if limitation:
            raise ProtocolError(limitation)
    else:
        listing = subprocess.check_output(["git", "ls-tree", revision, "--", "kernel.py"], cwd=workspace)
        if not listing.startswith(b"100644 blob "):
            raise ProtocolError("native revision kernel.py mode must be 0644")
        mode = 0o644
        # Read metadata from the same revision, never from a mutable worktree.
        metadata = subprocess.run(["git", "show", revision + ":solution.json"], cwd=workspace, capture_output=True)
        if metadata.returncode == 0:
            import json
            entries = json.loads(metadata.stdout).get("sources", [])
            if not isinstance(entries, list) or any(
                    (entry if isinstance(entry, str) else entry.get("path")) != "kernel.py" for entry in entries):
                raise ProtocolError("candidate declaration includes auxiliary sources")
    if mode != 0o644:
        raise ProtocolError("native kernel.py mode must be 0644")
    manifest = source_manifest(data)  # KS owns UTF-8 and canonical Source rules.
    if expected is not None and digest(manifest) != expected["sha256"]:
        raise ProtocolError("native candidate differs from selected public C")
    reference = publication.put_source("candidate", data)
    identity = candidate_workspace.source_identity(source)
    return SourceBinding(reference, identity, sha256(data), len(data))
