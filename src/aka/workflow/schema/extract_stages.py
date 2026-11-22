"""AKA's serial P1–P6 extraction owner, with immutable native-role receipts.

The small stage/plan/projection checks are adapted from the frozen upstream
artifacts.py (see NOTICE.md). No upstream Store, runner, CLI or scheduler is used.
The caller owns admission and the operation lock, as for Planner.design().
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import difflib
from pathlib import Path
import re
import secrets
from typing import Mapping
from urllib.parse import unquote, urlsplit

from aka.core.filesystem import durable_write_json
from .language import (CapacityError, CheckPolicy, SchemaError, check_design, json_material,
                       normalize_design, rules, text_policy, validate_documents)
from .language import yaml_source
from .language.rules import dispatch, values
from .language.normalization import loads_for_normalization
from .materials import (DesignRef, _directory, _json, closure_digest, publish_design,
                        read_design, role_material_view)
from .planner import (PROMPTS, Planner, PlannerResult, _DATA_ERRORS, _diagnostics,
                      _capture, _digest, _load, _loads, _write_files)
from .run_state import RecoveryRequired

STAGES = ("P1", "P2", "P3", "P4", "P5", "P6")
POLICY = CheckPolicy(skip_hardware=True, style="strict")
FIELDS = {
    "P1": ("META_DATA.Name", "META_DATA.Describe", "META_DATA.Source",
           "CONTRACT.Algorithm_requirement", "CONTRACT.Hardware_specification.Execution"),
    "P2": ("META_DATA.Source", "META_DATA.Dispatch"),
    "P3": ("DESIGN", "REALIZATION", "META_DATA.Dispatch.when"),
    "P5": ("DESIGN", "REALIZATION", "META_DATA.Source", "META_DATA.Dispatch.when"),
}
OBLIGATIONS = {
    "P1": ("algorithm", "io", "domain", "comparators", "execution", "source_gaps"),
    "P2": ("selection", "mechanisms", "instances", "directory"),
    "P3": ("root_complete", "source_fidelity", "contract", "dispatch_when"),
    "P4": ("coverage", "acyclic", "providers", "writing_order"),
    "P5": ("node_complete", "source_fidelity", "contract", "providers", "consumer_context"),
    "P6": ("complete", "source_fidelity", "approvals", "coverage", "unresolved", "knowledge"),
}
SOURCE_OBLIGATIONS = {"algorithm": ("reference", "implementation"), "io": ("io",),
    "domain": ("scope", "input"), "comparators": ("acceptance",), "execution": ("scope", "implementation"),
    "instances": ("implementation",), "source_fidelity": ("implementation",)}


@dataclass(frozen=True)
class ExtractRequest:
    prepared_id: str
    operation_id: str
    version: int
    directory: Path
    public_task: str
    target_hardware: str
    descriptor: object  # SourceDescriptor; imported at the capability boundary.
    target_gpu: Mapping
    source_snapshot: Mapping | None = None


@dataclass(frozen=True)
class ExtractConfig:
    max_repairs: int = 1
    max_revisions: int = 3
    normalize: bool = True

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in (self.max_repairs, self.max_revisions)):
            raise ValueError("extraction repair and upstream-revision allowances must be finite nonnegative integers")
        if type(self.normalize) is not bool:
            raise ValueError("normalize must be boolean")


def _semantic(value):
    # Distinguish scalar types and exact decimals, rather than Python's 1 == True.
    def typed(item):
        if isinstance(item, dict):
            return ["mapping", sorted((key, typed(child)) for key, child in item.items())]
        if isinstance(item, list):
            return ["sequence", [typed(child) for child in item]]
        return [type(item).__name__, json_material(item)]
    return _digest(_json(typed(value)))


def _manifest(files):
    return {name: _digest(raw) for name, raw in sorted(files.items())}


def _call_evidence(directory):
    attempt = Path(directory) / "attempts/1"
    collection = _load(attempt / "collected.json")
    files, _, problems = _capture(attempt / "raw")
    if problems or _manifest(files) != collection["files"]:
        raise ValueError("historical native-role raw output changed")
    return {"collection": _digest(yaml_source.read_source_bytes(attempt / "collected.json")),
            "request": {name: _digest(yaml_source.read_source_bytes(Path(directory) / name))
                        for name in ("intent.json", "attempts/1/started.json", "attempts/1/prompt.md")},
            "native": {name: _digest(yaml_source.read_source_bytes(attempt / "native" / name))
                       for name in ("receipt.json", "stdout.txt", "stderr.txt")}}


def _snapshot(directory, files):
    directory = _directory(directory)
    if directory.exists():
        for path in directory.iterdir():
            if path.name not in files or yaml_source.read_source_bytes(path) != files[path.name]:
                raise RecoveryRequired("retained stage snapshot conflicts with its original bytes; do not reauthor")
    _write_files(directory, files)
    reference = {"directory": str(directory), "files": _manifest(files)}
    if _read_snapshot(reference) != files:
        raise ValueError("stage snapshot changed while retaining it")
    return reference


def _read_snapshot(reference):
    directory = _directory(reference["directory"])
    if set(p.name for p in directory.iterdir()) != set(reference["files"]):
        raise ValueError("retained extraction snapshot inventory changed")
    files = {name: yaml_source.read_source_bytes(directory / name) for name in reference["files"]}
    if _manifest(files) != reference["files"]:
        raise ValueError("retained extraction snapshot bytes changed")
    return files


def _documents(files, *, normalizing=False):
    result = {}
    for name, raw in files.items():
        if name == "dispatch.md":
            continue
        if Path(name).name != name or not re.fullmatch(r"[a-z_][a-z0-9_]*\.yaml", name):
            raise ValueError("stage artifacts must be flat <node>.yaml files and dispatch.md")
        if normalizing:
            result[name[:-5]] = loads_for_normalization(raw.decode(), name[:-5], path=name,
                                                        skip_hardware=True).document
        else:
            result[name[:-5]] = yaml_source.loads(raw.decode(), name[:-5], path=name)
        value = result[name[:-5]].value
        if not isinstance(value, dict) or not isinstance(value.get("META_DATA"), dict):
            raise ValueError("each stage YAML requires an object and META_DATA mapping")
        if name == "root.yaml" and not isinstance(value.get("CONTRACT"), dict):
            raise ValueError("root requires the shared CONTRACT mapping")
    if "root" not in result or len(result) > 10:
        raise ValueError("a stage requires root and at most ten nodes")
    return result


def validate_plan(stage, plan, *, selection=None):
    if stage not in {"P2", "P4"} or not isinstance(plan, list):
        raise ValueError("a selection/writing plan is a list for P2/P4")
    entries = {}
    for row in plan:
        expected = {"id", "instance"} if stage == "P2" else {"id", "wait_for"}
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError("plan entry fields must be " + "/".join(sorted(expected)))
        errors = []
        values.node_id(row["id"], "plan.id", errors)
        if errors or row["id"] in entries:
            raise ValueError("plan node IDs must be valid and unique")
        entries[row["id"]] = row
    if stage == "P2":
        if "root" not in entries or not 1 <= len(entries) <= 10:
            raise ValueError("selection requires root and at most ten nodes")
        instances = [row["instance"] for row in plan]
        if any(not isinstance(i, str) or not i.strip() for i in instances) or len(set(instances)) != len(instances):
            raise ValueError("selected concrete source instances must be nonempty and unique")
    else:
        validate_plan("P2", selection)
        selected = {row["id"] for row in selection}
        if set(entries) != selected - {"root"}:
            raise ValueError("writing plan must cover every selected non-root node exactly once")
        for node, row in entries.items():
            waits = row["wait_for"]
            if (not isinstance(waits, list) or any(not isinstance(i, str) for i in waits)
                    or len(set(waits)) != len(waits) or not set(waits) <= selected or node in waits):
                raise ValueError("writing dependencies must be unique selected nodes without self-dependency")
        pending = {node: set(row["wait_for"]) - {"root"} for node, row in entries.items()}
        while pending:
            ready = {node for node, waits in pending.items() if not waits}
            if not ready:
                raise ValueError("writing plan contains a cycle")
            pending = {node: waits - ready for node, waits in pending.items() if node not in ready}
    return _semantic(entries)


def validate_index(files, selection, *, final=False):
    validate_plan("P2", selection)
    selected = {row["id"] for row in selection}
    available = {name[:-5] for name in files if name.endswith(".yaml")}
    if not available <= selected or final and available != selected:
        raise ValueError("actual YAML inventory differs from selected nodes")
    text = files["dispatch.md"].decode()
    lines = list(dispatch._visible(text))
    registered = set()
    for _, target in dispatch._links(lines):
        parsed = urlsplit(target)
        if not parsed.scheme and not parsed.netloc and unquote(parsed.path).endswith(".yaml"):
            name = unquote(parsed.path)
            if name not in files:
                raise ValueError("dispatch links to a missing YAML file")
            registered.add(name[:-5])
    design_lines, inside = [], False
    for number, line in lines:
        heading = dispatch._HEADING.fullmatch(line)
        if heading and len(heading[1]) <= 2:
            inside = len(heading[1]) == 2 and heading[2] == "完整设计"
        elif inside:
            design_lines.append((number, line))
    planned = set()
    for _, cell in dispatch._design_rows(design_lines)[1]:
        if "仅计划" in cell:
            match = re.fullmatch(r"仅计划 ([a-z_][a-z0-9_]*)\.yaml", cell)
            if not match or match[1] not in selected or match[1] in planned:
                raise ValueError("planned design cell must be exactly: 仅计划 <id>.yaml")
            planned.add(match[1])
    if "root" not in registered or planned & registered or planned | registered != selected:
        raise ValueError("dispatch must register root and exactly cover selected nodes with links or planned rows")
    if final and planned:
        raise ValueError("final directory contains an unfinished planned node")
    errors = dispatch.validate(text, {node: Path(node + ".yaml") for node in registered})
    if errors:
        raise SchemaError("; ".join(map(str, errors)), diagnostics=errors)
    return {"registered": sorted(registered), "planned": sorted(planned), "available": sorted(available)}


def _erase(value, fields):
    value = deepcopy(value)
    for path in (*fields, "META_DATA.Version"):
        owner = value
        parents = []
        for part in path.split(".")[:-1]:
            if not isinstance(owner, dict) or part not in owner:
                break
            parents.append((owner, part))
            owner = owner[part]
        else:
            if isinstance(owner, dict):
                owner.pop(path.split(".")[-1], None)
                for parent, key in reversed(parents):
                    if parent[key] == {}:
                        parent.pop(key)
                    else:
                        break
    return value


def check_scope(stage, before, after, *, node="root", normalizing=False):
    """Raw file additions/deletions and field permissions, independently of checks."""
    old, new = _documents(before, normalizing=normalizing), _documents(after, normalizing=normalizing)
    writes = {"P1": {"root.yaml"}, "P2": {"root.yaml", "dispatch.md"}, "P3": {"root.yaml"},
              "P4": {"dispatch.md"}, "P5": {node + ".yaml"}, "P6": set()}
    additions = {"P2": {"dispatch.md"}, "P5": {node + ".yaml"}}.get(stage, set())
    changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    if stage not in writes or changed - writes[stage] or not before.keys() <= after.keys() or after.keys() - before.keys() - additions:
        raise ValueError("write/addition/deletion outside " + stage + " scope: " + ", ".join(sorted(changed)))
    for key, document in new.items():
        current = document.value
        metadata = current.get("META_DATA", {})
        if not isinstance(metadata, dict) or type(metadata.get("Version")) is not int or metadata["Version"] < 1:
            raise ValueError("every node needs a positive integer Version")
        dispatch_value = metadata.get("Dispatch", {})
        if not isinstance(dispatch_value, dict) or dispatch_value.get("id", "root") != key:
            raise ValueError("node Dispatch.id differs from its filename")
        if key != "root" and ("CONTRACT" in current or any(
                metadata.get(field) != new["root"].value["META_DATA"].get(field) for field in ("Name", "Describe"))):
            raise ValueError("non-root nodes share Name/Describe and must not contain CONTRACT")
        if key in old:
            previous = old[key].value
            if key + ".yaml" in changed:
                fields = FIELDS.get(stage, ())
                if _semantic(_erase(previous, fields)) != _semantic(_erase(current, fields)):
                    raise ValueError(stage + " changed protected fields")
                previous_version = previous["META_DATA"]["Version"]
                if metadata["Version"] < previous_version or (_semantic(_erase(previous, ())) != _semantic(_erase(current, ()))
                        and metadata["Version"] <= previous_version):
                    raise ValueError("changed node content requires an increased Version")
        elif stage != "P5" or key != node or key == "root" or metadata["Version"] != 1:
            raise ValueError("only P5 may create its selected non-root node at Version 1")
        elif set(current) - {"META_DATA", "DESIGN", "REALIZATION"} or set(metadata) - {"Name", "Describe", "Version", "Source", "Dispatch"}:
            raise ValueError("new node contains protected/unknown fields")
    return sorted(changed)


def check_stage_snapshot(stage, files, *, selection=None, plan=None, node="root"):
    documents = _documents(files)
    root = documents["root"].value
    if stage != "P1":
        validate_index(files, selection, final=stage == "P6")
    if stage in {"P1", "P2"}:
        errors = rules.validate_structure({"root": root}, skip_hardware=True, allow_from=True)
        errors = [e for e in errors if not (e.path == "root.DESIGN" or e.path.startswith("root.DESIGN."))
                  and not (stage == "P2" and e.path == "root.META_DATA.Dispatch.when")]
        errors.extend(rules.validate_algorithm(root.get("CONTRACT", {}).get("Algorithm_requirement")))
        errors.extend(text_policy.check_values({k: v for k, v in root.items() if k not in {"CONTRACT", "DESIGN"}},
                                               skip_hardware=True))
        values.named(values.known_text, key=values.hardware_identity)(
            root.get("CONTRACT", {}).get("Hardware_specification", {}).get("Execution"),
            "root.CONTRACT.Hardware_specification.Execution", errors)
        if any(e.severity == "error" for e in errors):
            raise SchemaError("; ".join(map(str, errors)), diagnostics=errors)
        return None
    if len(selection) > 1 and not root["META_DATA"].get("Dispatch", {}).get("when"):
        raise ValueError("multiple selected nodes require root's actual Dispatch.when (defined by P3)")
    paths = {name: Path(name + ".yaml") for name in documents}
    resolved = validate_documents(paths, documents, policy=POLICY)
    if stage in {"P4", "P5", "P6"}:
        validate_plan("P4", plan, selection=selection)
    if stage in {"P5", "P6"}:
        entries = {row["id"]: row for row in plan}
        for name in ([node] if stage == "P5" else entries):
            actual = set(resolved.dependency_nodes[name]) - {name, "root"}
            if not actual <= set(entries[name]["wait_for"]):
                raise ValueError("actual from providers are absent from the declared writing dependencies")
    if stage == "P6":
        check_design(files, policy=POLICY)
    return resolved


def approval_projection(stage, files, *, selection=None, plan=None, node="root"):
    root = _documents(files)["root"].value
    if stage == "P1":
        subject = {"Name": root["META_DATA"]["Name"], "Describe": root["META_DATA"]["Describe"], "CONTRACT": root["CONTRACT"]}
    elif stage == "P2":
        subject = {"selection": validate_plan("P2", selection), "Source": root["META_DATA"].get("Source"),
                   "id": root["META_DATA"].get("Dispatch", {}).get("id", "root")}
    elif stage == "P3":
        subject = {"DESIGN": root["DESIGN"], "REALIZATION": root.get("REALIZATION"),
                   "when": root["META_DATA"].get("Dispatch", {}).get("when")}
    elif stage == "P4":
        subject = {"selection": validate_plan("P2", selection), "writing": validate_plan("P4", plan, selection=selection)}
    elif stage == "P5":
        docs = _documents(files)
        resolved = validate_documents({key: Path(key + ".yaml") for key in docs}, docs, policy=POLICY)
        subject = {"effective": resolved.effective_document(node), "origins": resolved.origins_dict(node),
                   "providers": {key: resolved.effective_document(key) for key in resolved.dependency_nodes[node] - {node}}}
    elif stage == "P6":
        subject = _manifest(files)
    else:
        raise ValueError("unknown extraction stage")
    return _semantic(subject)


def defining_stage(path):
    """Route a requested correction to the owner of that definition."""
    if path == "selection" or path.startswith("root.META_DATA.Source") or path == "root.META_DATA.Dispatch.id":
        return "P2", "root"
    if path in {"dispatch.md", "writing_plan"}:
        return "P4", "root"
    node = path.split(".", 1)[0]
    if node != "root":
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", node):
            raise ValueError("invalid correction node")
        return "P5", node
    if path.startswith(("root.CONTRACT", "root.META_DATA.Name", "root.META_DATA.Describe")):
        return "P1", "root"
    if path.startswith(("root.DESIGN", "root.REALIZATION", "root.META_DATA.Dispatch.when")):
        return "P3", "root"
    raise ValueError("revision must identify a supported defining field")


def _register(files, node):
    """Host-derived directory transition after a P5 approval; retained separately."""
    lines, count = [], 0
    for line in files["dispatch.md"].decode().splitlines(keepends=True):
        cells = dispatch._cells(line)
        if cells and cells[0] == "仅计划 " + node + ".yaml":
            line = line.replace(cells[0], f"[{node}]({node}.yaml)", 1)
            count += 1
        lines.append(line)
    if count != 1:
        raise ValueError("approved node must have exactly one pending directory row")
    return {**files, "dispatch.md": "".join(lines).encode()}


class Extractor:
    """Serial extraction capability borrowing an already configured AKA Planner."""
    def __init__(self, planner: Planner, *, config: ExtractConfig | None = None):
        self.planner, self.config = planner, config or ExtractConfig()

    def _base_inputs(self, request, sources):
        inputs = role_material_view(self.planner.materials_snapshot)
        inputs.update(sources.files)
        inputs.update({"task/public.md": request.public_task.encode(), "task/hardware.md": request.target_hardware.encode()})
        return inputs

    def _quotes(self, quotes, inputs, sources):
        if not isinstance(quotes, list) or not quotes or len(quotes) > 100:
            raise ValueError("evidence requires a nonempty bounded list of exact quotes")
        for quote in quotes:
            if not isinstance(quote, dict) or set(quote) != {"path", "line_start", "line_end", "quote", "sha256"}:
                raise ValueError("quote requires path/line_start/line_end/quote/sha256")
            name = quote["path"]
            if not isinstance(name, str) or name not in inputs:
                raise ValueError("quote is outside the frozen role view")
            start, end = quote["line_start"], quote["line_end"]
            lines = inputs[name].decode().splitlines()
            if (type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines)
                    or end - start >= 200 or quote["sha256"] != _digest(inputs[name])
                    or quote["quote"] != "\n".join(lines[start - 1:end])):
                raise ValueError("evidence quote or digest differs from its exact frozen source lines")
            if name.startswith("sources/") and name != "sources/manifest.json":
                sources.verify_quote({**quote, "path": name.removeprefix("sources/")})
        return quotes

    @staticmethod
    def _knowledge(value, *, required=False):
        if not isinstance(value, list) or len(value) > 100:
            raise ValueError("knowledge destinations must be a bounded list")
        for item in value:
            if not isinstance(item, dict) or set(item) != {"finding", "destination", "rationale"} or any(
                    not isinstance(v, str) or not v.strip() for v in item.values()):
                raise ValueError("knowledge items require finding/destination/rationale")
        if required and not value:
            raise ValueError("P6 must explicitly account for knowledge destinations, including none with a reason")

    def _report(self, files, binding, *, reviewer, inputs, sources, obligations=()):
        if "decision.json" not in files or any(name != "decision.json" and not name.startswith("design/") for name in files):
            raise ValueError("role output requires decision.json and only its permitted complete design")
        if len(files["decision.json"]) > 131072:
            raise ValueError("stage receipt exceeds 128 KiB")
        report = _loads(files["decision.json"])
        expected = {"binding", "decision", "checks", "revision", "knowledge"} if reviewer else {
            "binding", "decision", "evidence", "selection", "writing_plan", "unresolved", "revision", "knowledge"}
        if not isinstance(report, dict) or set(report) != expected or report["binding"] != binding:
            raise ValueError("stage receipt has wrong fields or stale operation/nonce/input/candidate binding")
        allowed = {"approve", "revise"} if reviewer else {"complete", "revise"}
        if not isinstance(report["decision"], str) or report["decision"] not in allowed:
            raise ValueError("invalid stage decision")
        revision = report["revision"]
        if report["decision"] == "revise":
            if not isinstance(revision, dict) or set(revision) != {"path", "reason"} or any(
                    not isinstance(v, str) or not v.strip() for v in revision.values()):
                raise ValueError("revise requires a concrete defining path and reason")
            defining_stage(revision["path"])
        elif revision is not None:
            raise ValueError("completed/approved receipt cannot contain a pending revision")
        self._knowledge(report["knowledge"], required=binding["stage"] == "P6" and report["decision"] != "revise")
        if reviewer:
            if any(name != "decision.json" for name in files):
                raise ValueError("reviewer design is read-only")
            checks = report["checks"]
            if not isinstance(checks, dict) or set(checks) != set(obligations):
                raise ValueError("reviewer must cover exactly every host-required conclusion")
            for key, item in checks.items():
                if (not isinstance(item, dict) or set(item) != {"status", "rationale", "evidence"}
                        or not isinstance(item["status"], str) or item["status"] not in {"pass", "fail", "unknown"}
                        or not isinstance(item["rationale"], str) or not item["rationale"].strip()):
                    raise ValueError("each conclusion requires pass/fail/unknown, rationale and exact evidence")
                self._quotes(item["evidence"], inputs, sources)
                if key in SOURCE_OBLIGATIONS:
                    roles = SOURCE_OBLIGATIONS[key]
                    eligible = [row for role in roles for row in sources.manifest["roles"].get(role, ())]
                    if eligible:
                        matches = [quote for quote in item["evidence"] if any(
                            quote["path"] == "sources/" + row["path"] and row["line_start"] <= quote["line_start"]
                            <= quote["line_end"] <= row["line_end"] for row in eligible)]
                        if not matches:
                            raise ValueError(key + " requires evidence from its actual frozen source role")
                    elif not any(quote["path"] == "sources/manifest.json" for quote in item["evidence"]):
                        raise ValueError("absent source role must cite its explicit frozen missing record")
                if key.startswith("comparator:"):
                    comparator = next(row for row in sources.manifest["comparators"] if row["output"] == key.split(":", 1)[1])
                    if comparator["rule"]:
                        for declaration in comparator["evidence"]:
                            if not any(quote["path"] == "sources/" + declaration["path"]
                                       and quote["line_start"] <= declaration["line_start"] <= declaration["line_end"] <= quote["line_end"]
                                       for quote in item["evidence"]):
                                raise ValueError(key + " must cite this output's actual acceptance declaration")
                    elif not any(quote["path"] == "sources/manifest.json" for quote in item["evidence"]):
                        raise ValueError(key + " must cite its explicit unavailable comparator record")
                if report["decision"] == "approve" and item["status"] != "pass":
                    raise ValueError("approval requires explicit pass on every required conclusion")
        else:
            self._quotes(report["evidence"], inputs, sources)
            if not any(quote["path"].startswith("sources/") for quote in report["evidence"]):
                raise ValueError("worker must cite the actual source snapshot")
            if not isinstance(report["unresolved"], list) or any(not isinstance(i, str) or not i.strip() for i in report["unresolved"]):
                raise ValueError("unresolved source findings must be an explicit list")
            if report["decision"] == "complete" and report["unresolved"]:
                raise ValueError("a complete stage cannot retain unresolved obligations")
        return report

    def _call(self, directory, inputs, binding, *, reviewer=False):
        name = "extract_reviewer.md" if reviewer else "extract_worker.md"
        prompt = (yaml_source.read_source_bytes(PROMPTS / "extract.md") + b"\n\n" +
                  yaml_source.read_source_bytes(PROMPTS / name)).decode()
        readonly = reviewer or binding["stage"] == "P6"
        replay = (directory / "attempts/1/collected.json").exists()
        try:
            receipt, files, problems = self.planner.retained_role(directory=directory, inputs=inputs, prompt=prompt,
                operation_id=binding["operation_id"] + ":" + binding["nonce"] + ":" + binding["role"],
                read_only_design=readonly, config=replace(self.planner.config, check_policy=POLICY))
        except _DATA_ERRORS as error:
            if replay:
                raise RecoveryRequired("retained extraction role evidence changed; settle it without reauthoring") from error
            raise
        if receipt is None:
            raise RecoveryRequired("Extraction call started without raw receipt; settle its original native owner")
        if receipt["timed_out"] or receipt["exit_status"] != 0:
            raise RecoveryRequired("Extraction native role failed/timed out; retained owner receipt requires diagnosis")
        if problems:
            raise ValueError("; ".join(problems))
        return files

    def _checked(self, directory, stage, before, raw, selection, plan, node):
        check_scope(stage, before, raw, node=node, normalizing=True)
        candidate = raw
        try:
            check_scope(stage, before, candidate, node=node)
            check_stage_snapshot(stage, candidate, selection=selection, plan=plan, node=node)
        except _DATA_ERRORS as original:
            if not self.config.normalize or stage not in {"P3", "P4", "P5"}:
                raise
            try:
                normalized = normalize_design(raw, skip_hardware=True)
                candidate = {**raw, **{key + ".yaml": value.encode() for key, value in normalized.candidates.items()}}
                check_scope(stage, before, candidate, node=node)
                check_stage_snapshot(stage, candidate, selection=selection, plan=plan, node=node)
            except _DATA_ERRORS:
                raise original
            durable_write_json(directory / "normalization.json", {
                "raw_digest": closure_digest(raw), "checked_digest": closure_digest(candidate),
                "corrections": [asdict(i) for i in normalized.corrections],
                "diffs": {name: "".join(difflib.unified_diff(raw[name].decode().splitlines(True),
                    content.decode().splitlines(True), fromfile="raw/" + name, tofile="checked/" + name))
                    for name, content in candidate.items() if content != raw[name]}})
        reference = _snapshot(directory / "checked", candidate)
        reread = _read_snapshot(reference)
        check_scope(stage, before, reread, node=node)
        check_stage_snapshot(stage, reread, selection=selection, plan=plan, node=node)
        return reference

    @staticmethod
    def _next(state):
        done = {(item["stage"], item["node"]) for item in state["approvals"]}
        for stage in STAGES[:4]:
            if (stage, "root") not in done:
                return stage, "root"
        for row in state["plan"]:
            if ("P5", row["id"]) not in done and all(
                    other == "root" or ("P5", other) in done for other in row["wait_for"]):
                return "P5", row["id"]
        if sum(stage == "P5" for stage, _ in done) != len(state["plan"]):
            raise ValueError("no ready writing node in validated plan")
        return "P6", "root"

    def _audit(self, state, current):
        """Recheck retained raw evidence, current projections and exact transitions."""
        previous = state["initial"]
        preceding = []
        for ref in state["approvals"]:
            approval = _load(Path(ref["path"]))
            if _digest(yaml_source.read_source_bytes(Path(ref["path"]))) != ref["identity"] or approval["predecessors"] != preceding:
                raise ValueError("approval receipt or its predecessor chain changed")
            if approval["before"] != previous:
                raise ValueError("stage conversion chain has a missing or altered predecessor")
            before, raw, checked = map(_read_snapshot, (approval["before"], approval["raw"], approval["checked"]))
            check_scope(approval["stage"], before, raw, node=approval["node"], normalizing=True)
            check_scope(approval["stage"], before, checked, node=approval["node"])
            check_stage_snapshot(approval["stage"], checked, selection=approval["selection"], plan=approval["plan"], node=approval["node"])
            if raw != checked:
                normalized = normalize_design(raw, skip_hardware=True)
                expected = {**raw, **{key + ".yaml": text.encode() for key, text in normalized.candidates.items()}}
                record = Path(ref["path"]).parent / "normalization.json"
                if checked != expected or _digest(yaml_source.read_source_bytes(record)) != approval["normalization_digest"]:
                    raise ValueError("normalization transformation differs from its original retained record")
            elif approval["normalization_digest"] is not None:
                raise ValueError("unchanged raw bytes cannot claim an unrelated normalization record")
            projection = approval_projection(approval["stage"], current, selection=state["selection"], plan=state["plan"], node=approval["node"])
            if projection != approval["projection"]:
                raise ValueError("an active approval no longer covers its defining fields/providers")
            for role in ("worker", "reviewer"):
                receipt = Path(approval[role + "_directory"]) / "attempts/1/raw/decision.json"
                if _digest(yaml_source.read_source_bytes(receipt)) != approval[role + "_digest"]:
                    raise ValueError("historical role receipt changed; cannot re-sign old evidence")
                if _call_evidence(approval[role + "_directory"]) != approval[role + "_call_evidence"]:
                    raise ValueError("historical native receipt/stdout/stderr or raw collection changed")
            after = _read_snapshot(approval["after"])
            expected = _register(checked, approval["node"]) if approval["stage"] == "P5" else checked
            if after != expected:
                raise ValueError("stage-to-current transformation differs from the documented host derivation")
            previous = approval["after"]
            preceding.append(ref["identity"])
        if _read_snapshot(previous) != current:
            raise ValueError("current extraction bytes differ from the active approval chain")

    def _revise(self, state, revision, stage, node):
        target = defining_stage(revision["path"])
        if STAGES.index(target[0]) > STAGES.index(stage):
            raise ValueError("revision cannot skip forward to an uncompleted stage")
        if target == (stage, node):
            return False
        indices = [i for i, row in enumerate(state["approvals"]) if (row["stage"], row["node"]) == target]
        if not indices:
            raise ValueError("revision must target an active defining-stage approval")
        if state["revisions"] >= self.config.max_revisions:
            raise ValueError("bounded upstream revision allowance exhausted")
        index = indices[0]
        first = _load(Path(state["approvals"][index]["path"]))
        state["revoked"].append({"reason": revision, "approvals": state["approvals"][index:]})
        state["approvals"] = state["approvals"][:index]
        state["current"] = first["before"]
        state["selection"], state["plan"] = first["previous_selection"], first["previous_plan"]
        state["revisions"] += 1
        state["failures"] = 0
        return True

    def extract(self, request: ExtractRequest) -> PlannerResult:
        from .extract_sources import (snapshot_sources, read_source_snapshot, validate_comparisons,
                                      validate_source_reference, validate_target_gpu, validate_source_request)
        if not isinstance(request, ExtractRequest) or not all(isinstance(i, str) and i.strip() for i in (
                request.prepared_id, request.operation_id, request.public_task, request.target_hardware)):
            raise ValueError("extraction needs explicit source, prepared identity, public Task and target hardware")
        if type(request.version) is not int or request.version < 1:
            raise ValueError("extraction version must be positive")
        directory = _directory(request.directory)
        # The independent device/source gate runs before any role, including replay.
        validate_target_gpu(request.target_gpu, request.target_hardware)
        sources = validate_source_request(read_source_snapshot(request.source_snapshot), request.descriptor, request.target_gpu) \
            if request.source_snapshot is not None else snapshot_sources(directory / "sources", request.descriptor, target_gpu=request.target_gpu)
        sources.require_same_gpu()
        sources = read_source_snapshot(sources.ref())
        base = self._base_inputs(request, sources)
        binding = {"prepared_id": request.prepared_id, "operation_id": request.operation_id,
                   "version": request.version, "source_identity": sources.identity,
                   "inputs": _manifest(base), "config": asdict(self.config),
                   "check_policy": POLICY.as_dict(),
                   "role_config": asdict(self.planner.config), "agent_identity": self.planner.agent.configuration_identity,
                   "prompts": {name: _digest(yaml_source.read_source_bytes(PROMPTS / name))
                               for name in ("extract.md", "extract_worker.md", "extract_reviewer.md")}}
        state_path = directory / "state.json"
        if state_path.exists():
            state = _load(state_path)
            if state["binding"] != binding:
                raise ValueError("extraction replay conflicts with frozen input, role configuration or prompts")
        else:
            initial = _snapshot(directory / "initial", {"root.yaml": base["schema-materials/starter.yaml"]})
            state = {"format": "aka.schema.extract.v1", "binding": binding, "source": sources.ref(),
                     "initial": initial, "current": initial, "selection": None, "plan": None,
                     "approvals": [], "revoked": [], "pending": None, "serial": 0,
                     "calls": 0, "revisions": 0, "failures": 0, "diagnostics": [], "result": None}
            durable_write_json(state_path, state)
        current = _read_snapshot(state["current"])
        self._audit(state, current)
        if state["result"] is not None:
            return PlannerResult.from_dict(state["result"])
        while not any(row["stage"] == "P6" for row in state["approvals"]):
            stage, node = self._next(state)
            if state["pending"] is None:
                state["serial"] += 1
                state["pending"] = {"stage": stage, "node": node, "serial": state["serial"], "nonce": secrets.token_hex(16)}
                durable_write_json(state_path, state)
            pending = state["pending"]
            if (pending["stage"], pending["node"]) != (stage, node):
                raise ValueError("pending stage conflicts with the active approval chain")
            step = directory / "stages" / str(pending["serial"])
            before = _read_snapshot(state["current"])
            inputs = {**base, **{"current-design/" + name: raw for name, raw in before.items()}}
            worker_binding = {"operation_id": request.operation_id, "prepared_id": request.prepared_id,
                "source_identity": sources.identity, "stage": stage, "node": node, "role": "worker",
                "nonce": pending["nonce"], "attempt": pending["serial"], "inputs_digest": _digest(_json(_manifest(base))),
                "candidate_digest": closure_digest(before), "projection_digest": None}
            context = {"binding": worker_binding, "selection": state["selection"], "writing_plan": state["plan"],
                       "required_checks": list(OBLIGATIONS[stage]) +
                           (["comparator:" + name for name in sources.manifest["outputs"]] if stage == "P1" else []) +
                           (["node:" + row["id"] for row in state["selection"]] if stage == "P6" else []),
                       "approvals": state["approvals"], "diagnostics": state["diagnostics"][-5:]}
            inputs["task/stage.json"] = _json(context)
            try:
                if stage == "P6":
                    check_stage_snapshot(stage, before, selection=state["selection"], plan=state["plan"])
                    self._audit(state, before)
                worker_files = self._call(step / "worker", inputs, worker_binding)
                worker = self._report(worker_files, worker_binding, reviewer=False, inputs=inputs, sources=sources)
                if worker["decision"] == "revise":
                    if any(name != "decision.json" for name in worker_files):
                        raise ValueError("a revision request cannot edit the design")
                    redirected = self._revise(state, worker["revision"], stage, node)
                    raise _Revision(worker["revision"], redirected)
                selection = worker["selection"] if stage == "P2" else state["selection"]
                plan = worker["writing_plan"] if stage == "P4" else state["plan"]
                if stage != "P2" and worker["selection"] != state["selection"] or stage != "P4" and worker["writing_plan"] != state["plan"]:
                    raise ValueError("worker changed a selection/writing plan outside its defining stage")
                if stage == "P2":
                    validate_plan(stage, selection)
                if stage == "P4":
                    validate_plan(stage, plan, selection=selection)
                raw = {name.removeprefix("design/"): content for name, content in worker_files.items() if name.startswith("design/")}
                if stage == "P6":
                    if raw:
                        raise ValueError("P6 is read-only; its worker cannot emit edited design files")
                    raw = before
                raw_ref = _snapshot(step / "raw", raw)
                checked_ref = self._checked(step, stage, before, raw, selection, plan, node)
                checked = _read_snapshot(checked_ref)
                if stage == "P1":
                    criteria = _documents(checked)["root"].value["CONTRACT"]["Algorithm_requirement"]["Golden_reference"]["Tolerance"]
                    validate_comparisons(sources, criteria)
                    validate_source_reference(sources, _documents(checked)["root"].value["CONTRACT"][
                        "Algorithm_requirement"]["Golden_reference"]["Source"], role="reference")
                if stage == "P2":
                    for row in selection:
                        validate_source_reference(sources, row["instance"], role="implementation")
                if stage in {"P2", "P3", "P5", "P6"}:
                    documents = _documents(checked)
                    selected = {row["id"]: row["instance"] for row in selection}
                    for name, document in documents.items():
                        source_ref = document.value["META_DATA"].get("Source")
                        validate_source_reference(sources, source_ref, role="implementation")
                        if source_ref != selected[name]:
                            raise ValueError("node source differs from its selected concrete instance")
                projection = approval_projection(stage, checked, selection=selection, plan=plan, node=node)
                review_binding = {**worker_binding, "role": "reviewer", "candidate_digest": closure_digest(checked),
                                  "projection_digest": projection}
                review_inputs = {**base, **{"candidate/" + name: content for name, content in checked.items()},
                    "task/stage.json": _json({**context, "binding": review_binding, "selection": selection, "writing_plan": plan}),
                    "task/worker.json": worker_files["decision.json"]}
                review_files = self._call(step / "reviewer", review_inputs, review_binding, reviewer=True)
                review = self._report(review_files, review_binding, reviewer=True, inputs=review_inputs, sources=sources,
                                      obligations=context["required_checks"])
                if review["decision"] == "revise":
                    redirected = self._revise(state, review["revision"], stage, node)
                    raise _Revision(review["revision"], redirected)
                after = _register(checked, node) if stage == "P5" else checked
                after_ref = _snapshot(step / "after", after)
                approval = {"format": "aka.schema.extract-approval.v1", "stage": stage, "node": node,
                    "binding": review_binding, "before": state["current"], "raw": raw_ref, "checked": checked_ref,
                    "after": after_ref, "projection": projection, "selection": selection, "plan": plan,
                    "previous_selection": state["selection"], "previous_plan": state["plan"],
                    "predecessors": [row["identity"] for row in state["approvals"]],
                    "worker_directory": str(step / "worker"), "reviewer_directory": str(step / "reviewer"),
                    "worker_digest": _digest(worker_files["decision.json"]), "reviewer_digest": _digest(review_files["decision.json"]),
                    "worker_call_evidence": _call_evidence(step / "worker"), "reviewer_call_evidence": _call_evidence(step / "reviewer"),
                    "normalization_digest": _digest(yaml_source.read_source_bytes(step / "normalization.json")) if raw != checked else None,
                    "judgment": "independent model judgment; not a formal proof"}
                durable_write_json(step / "approval.json", approval)
                state["approvals"].append({"stage": stage, "node": node, "path": str(step / "approval.json"),
                                          "identity": _digest(yaml_source.read_source_bytes(step / "approval.json"))})
                state["current"], state["selection"], state["plan"] = after_ref, selection, plan
                state["failures"] = 0
            except (ValueError, SchemaError, UnicodeError, CapacityError, yaml_source.ReadError) as error:
                if isinstance(error, _Revision):
                    state["diagnostics"].append({"category": "extract.revision", "message": str(error)})
                else:
                    state["diagnostics"].extend(_diagnostics(error))
                if not isinstance(error, _Revision) or not error.redirected:
                    state["failures"] += 1
                durable_write_json(step / "rejected.json", {"diagnostics": state["diagnostics"][-5:], "stage": stage, "node": node})
                if state["failures"] > self.config.max_repairs:
                    state["result"] = PlannerResult("failed", None, request.operation_id, directory,
                        tuple(state["diagnostics"]), calls=self._calls(directory)).as_dict()
            state["calls"] = self._calls(directory)
            state["pending"] = None
            durable_write_json(state_path, state)
            if state["result"] is not None:
                return PlannerResult.from_dict(state["result"])
        current = _read_snapshot(state["current"])
        self._audit(state, current)
        publication, number = directory / "published", 0
        while publication.exists() and any(publication.iterdir()) and not (publication / "manifest.json").exists():
            number += 1
            publication = directory / ("published-recovery-" + str(number))
        ref = publish_design(publication, current, version=request.version, prepared_id=request.prepared_id,
            operation_id=request.operation_id, materials_snapshot=self.planner.materials_snapshot,
            mode="extract", policy=POLICY, approval_refs=state["approvals"])
        read_design(ref)
        result = PlannerResult("replace", ref, request.operation_id, directory, tuple(state["diagnostics"]),
            rationale="Complete same-GPU extraction passed P1–P6 with independent stage reviews",
            hypothesis_key="extract:" + closure_digest(current), changed_execution_decisions=("Extracted the selected source mechanisms",),
            next_actions=("Implement the frozen extracted design using native correctness and measurement",), calls=self._calls(directory))
        state["result"] = result.as_dict()
        durable_write_json(state_path, state)
        return result

    @staticmethod
    def _calls(directory):
        return len(list((directory / "stages").glob("*/[wr]*/attempts/1/started.json")))


class _Revision(ValueError):
    def __init__(self, revision, redirected):
        super().__init__(revision)
        self.redirected = redirected


def extract(request: ExtractRequest, *, planner: Planner, config: ExtractConfig | None = None):
    return Extractor(planner, config=config).extract(request)
