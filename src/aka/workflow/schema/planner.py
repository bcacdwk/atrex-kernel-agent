"""AKA's bounded design/optimize writer and scoped, fresh-session role calls.

The controller owns the operation lock, admission, budgets and DesignRef root
registration. This module retains each actual call and output, never launches a
CLI directly, and never interprets a completed message as a valid design.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import difflib
import hashlib
import json
from pathlib import Path
import stat
import time
from typing import Callable, Mapping

from aka.contracts.agent import (AgentAccess, AgentMount, AgentToolBinding, NativeInvocation,
                                 AgentRuntimeCapabilities, NormalizedAgentEvent, TokenUsage)
from aka.contracts.agent_resources import ResourcesSnapshot
from aka.contracts.workspace import RevisionContent
from aka.core.filesystem import durable_write_json, durable_write_text, ensure_private_directory

from .language import (CapacityError, CheckPolicy, SchemaError, check_design,
                       collect_design, effective_yaml, json_material, normalize_design)
from .language.yaml_source import ReadError, read_source_bytes
from .materials import (DesignRef, _directory, _json, _member, closure_digest,
                        publish_design, read_design, role_material_view)

PROMPTS = Path(__file__).parent / "prompts"
MAX_FEEDBACK_BYTES = 32_768
MAX_DECISION_BYTES = 131_072
_DATA_ERRORS = (ValueError, OSError, UnicodeError, SchemaError, CapacityError, ReadError)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load(path):
    return _loads(read_source_bytes(path))


def _loads(raw):
    def unique(items):
        result = {}
        for name, value in items:
            if name in result:
                raise ValueError("duplicate JSON field: " + name)
            result[name] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=unique,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except RecursionError as error:
        raise CapacityError("JSON nesting exceeds the parser's bounded capacity") from error


def _write_files(directory, files):
    ensure_private_directory(directory)
    for name, raw in sorted(files.items()):
        relative = _member(name)
        durable_write_text(directory / relative, raw.decode("utf-8"), mode=0o444)


def _diagnostics(error):
    details = getattr(error, "diagnostics", ())
    return tuple(asdict(item) for item in details) or ({
        "category": "capacity" if isinstance(error, CapacityError) else "writer.output",
        "severity": "error", "path": "output", "message": str(error)[:12_000]},)


def _notice(category, message):
    return {"category": category, "severity": "warning", "path": "output", "message": message}


@dataclass(frozen=True)
class PlannerConfig:
    agent_cli: str = "claude"
    timeout_seconds: float = 3600
    reasoning_effort: str = "max"
    max_repairs: int = 1
    normalize: bool = True
    check_policy: CheckPolicy = field(default_factory=CheckPolicy)

    def __post_init__(self):
        from aka.agent.provider import validate_timeout
        validate_timeout(self.timeout_seconds)
        if self.timeout_seconds is None or type(self.max_repairs) is not int or self.max_repairs < 0:
            raise ValueError("writer needs a finite timeout and an explicit nonnegative repair-call limit")
        if self.reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("unsupported reasoning effort")
        if type(self.normalize) is not bool or not isinstance(self.check_policy, CheckPolicy):
            raise ValueError("invalid writer check policy")


@dataclass(frozen=True)
class PlannerRequest:
    prepared_id: str
    operation_id: str
    version: int
    directory: Path
    public_task: str
    target_hardware: str
    current_design: DesignRef | None = None
    feedback: tuple[dict, ...] = ()
    require_new: bool = False
    source_files: Mapping[str, bytes] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannerResult:
    status: str
    design_ref: DesignRef | None
    operation_id: str
    directory: Path
    diagnostics: tuple[dict, ...] = ()
    rationale: str = ""
    changes: tuple[dict, ...] = ()
    evidence: tuple[dict, ...] = ()
    next_actions: tuple[str, ...] = ()
    hypothesis_key: str = ""
    changed_execution_decisions: tuple[str, ...] = ()
    calls: int = 0

    def as_dict(self):
        value = asdict(self)
        value["directory"] = str(self.directory)
        value["design_ref"] = self.design_ref.as_dict() if self.design_ref else None
        return value

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["directory"] = Path(value["directory"])
        if value["design_ref"]:
            value["design_ref"] = DesignRef.from_dict(value["design_ref"])
            read_design(value["design_ref"])
        for key in ("diagnostics", "changes", "evidence", "next_actions", "changed_execution_decisions"):
            value[key] = tuple(value[key])
        return cls(**value)


class _ReadTool:
    """A role can inspect only the in-memory frozen view or its candidate closure."""
    def __init__(self, name, workspace, inputs, policy):
        self.name, self.workspace, self.inputs, self.policy = name, workspace, inputs, policy

    def handle(self, payload, context):
        try:
            if context.workspace != self.workspace or len(payload) > 16_384:
                raise ValueError("invalid Schema tool scope or request size")
            request = _loads(payload)
            if not isinstance(request, dict):
                raise ValueError("Schema tool request must be an object")
            if self.name == "schema.read_evidence":
                if set(request) - {"source", "start", "end"}:
                    raise ValueError("unknown evidence request fields")
                source = request.get("source")
                if not isinstance(source, str) or source not in self.inputs:
                    raise ValueError("source is outside the frozen role view")
                raw = self.inputs[source]
                lines = raw.decode("utf-8").splitlines()
                start, end = request.get("start", 1), request.get("end", min(len(lines), 200))
                if (type(start) is not int or type(end) is not int
                        or not 1 <= start <= end <= len(lines) or end - start >= 200):
                    raise ValueError("evidence requires an existing one-based range of at most 200 lines")
                return _json({"source": source, "sha256": _digest(raw), "start": start,
                              "end": end, "text": "\n".join(lines[start - 1:end])})
            if set(request) - {"source", "node"}:
                raise ValueError("unknown check request fields")
            source = request.get("source", "candidate")
            if source == "candidate":
                path = _directory(self.workspace / "output/design")
                checked = check_design(path, policy=self.policy)
            elif source == "current":
                files = {name.removeprefix("current-design/"): raw for name, raw in self.inputs.items()
                         if name.startswith("current-design/")}
                checked = check_design(files, policy=self.policy)
            else:
                raise ValueError("check source must be candidate or current")
            result = {"valid": True, "closure_digest": closure_digest(checked.files),
                      "diagnostics": [asdict(item) for item in checked.diagnostics]}
            if self.name == "schema.expand":
                node = request.get("node", "root")
                if not isinstance(node, str) or node not in checked.resolved.raw_task:
                    raise ValueError("unknown design node")
                result["yaml"] = effective_yaml(checked.resolved, node)
            return _json(result)
        except _DATA_ERRORS as error:
            return _json({"valid": False, "diagnostics": _diagnostics(error)})


def invoke_role(agent, *, workspace: Path, observations_directory: Path,
                inputs: Mapping[str, bytes], prompt: str, config: PlannerConfig,
                operation_id: str, observe_result: Callable | None = None,
                read_only_design: bool = False) -> NativeInvocation:
    """One fresh native session, reusable by extraction workers and reviewers.

    Call under the controller's Agent invocation_scope. No retry or admission
    fallback is performed here; a reviewer always gets a new session identity.
    Inputs are the caller's explicit public view, never a host directory mount.
    operation_id identifies this physical call, including its owner's persistent
    stage/attempt suffix; observe_result must deduplicate that call, not sessions.
    """
    if not agent.capabilities.filesystem_isolation:
        raise ValueError("Schema roles require an Agent with filesystem isolation")
    workspace, observations_directory = _directory(workspace), _directory(observations_directory)
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError("a fresh role session requires an empty workspace")
    inputs = dict(inputs)
    inputs["schema_call.py"] = read_source_bytes(PROMPTS / "schema_call.py")
    _write_files(workspace / "inputs", inputs)
    ensure_private_directory(workspace / "output/design")
    ensure_private_directory(observations_directory)
    access = AgentAccess(mounts=(AgentMount(workspace, workspace, True, "Schema role workspace"),),
                         read_only_paths=(workspace / "inputs",) +
                         ((workspace / "output/design",) if read_only_design else ()))
    bindings = tuple(AgentToolBinding(name, _ReadTool(name, workspace, inputs, config.check_policy))
                     for name in ("schema.check", "schema.expand", "schema.read_evidence"))
    with agent.tool_session(workspace, access=access, bindings=bindings,
                            deadline=time.monotonic() + config.timeout_seconds) as grant:
        environment = {**grant.environment, "ATREX_AGENT_TIMEOUT_SECONDS": str(config.timeout_seconds)}
        combined = AgentAccess(mounts=(*access.mounts, *grant.filesystem_access.mounts),
                               read_only_paths=access.read_only_paths)
        session = agent.open_session(config.agent_cli, workspace, environment=environment,
                                     output_directory=observations_directory, access=combined)
        def retain(result):
            # The native owner's existing optional recorder runs before derived
            # public observation export, whose failure must not lose this result.
            receipt = asdict(result)
            receipt.pop("stdout")
            receipt.pop("stderr")
            path = observations_directory / "receipt.json"
            if path.exists() and _load(path) == _loads(_json(receipt)):
                return
            durable_write_text(observations_directory / "stdout.txt", result.stdout, mode=0o444)
            durable_write_text(observations_directory / "stderr.txt", result.stderr, mode=0o444)
            durable_write_json(path, receipt)
        from aka.agent.native import NativeSession
        if isinstance(session, NativeSession):
            session.record_native_invocation = retain
        try:
            result = session.invoke(prompt, reasoning_effort=config.reasoning_effort)
            retain(result)
            if observe_result is not None:
                observe_result(operation_id, result)
            return result
        finally:
            session.close()


def _capture(directory):
    """Retain complete bounded output, including invalid extra-file diagnostics."""
    files, inventory, problems = {}, [], []
    total = 0
    def walk(root, prefix=""):
        nonlocal total
        if root.resolve() != root or root.is_symlink():
            raise ValueError("output directory traverses a link")
        for path in sorted(root.iterdir()):
            name = prefix + path.name
            _member(name)
            if len(inventory) >= 200:
                raise CapacityError("writer output exceeds 200 entries")
            mode = path.lstat().st_mode
            inventory.append(name)
            if stat.S_ISDIR(mode):
                if name != "design":
                    problems.append("unexpected output subdirectory: " + name)
                if len(Path(name).parts) > 4:
                    raise CapacityError("writer output directory nesting exceeds four levels")
                walk(path, name + "/")
            elif stat.S_ISREG(mode) and path.lstat().st_nlink == 1:
                raw = read_source_bytes(path)
                total += len(raw)
                if total > 180 * 1024 * 1024:
                    raise CapacityError("writer output exceeds 180 MiB")
                raw.decode("utf-8")
                files[name] = raw
            else:
                problems.append("output contains a link or nonregular file: " + name)
    directory = _directory(directory)
    if not directory.is_dir():
        raise ValueError("writer removed its output directory")
    walk(directory)
    return files, inventory, problems


def _text_envelope(stdout):
    """Extract the final textual response, never tool inputs or arbitrary YAML."""
    final = None
    for line in stdout.splitlines():
        try:
            event = _loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "result" and isinstance(event.get("result"), str) and event["result"].strip():
            final = event["result"]
        elif event.get("type") == "assistant":
            message = event.get("message")
            content = message.get("content", []) if isinstance(message, dict) else []
            if not isinstance(content, list):
                continue
            parts = [part["text"] for part in content if isinstance(part, dict)
                     and part.get("type") == "text" and isinstance(part.get("text"), str)]
            if parts:
                final = "\n".join(parts)
        elif event.get("type") == "item.completed":
            item = event.get("item", {})
            if isinstance(item, dict) and item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                final = item["text"]
    final = stdout.strip() if final is None else final.strip()
    if final.startswith("```json\n") and final.endswith("\n```"):
        final = final[8:-4]
    value = _loads(final)
    if not isinstance(value, dict) or set(value) != {"decision", "files"} or not isinstance(value["files"], dict):
        raise ValueError("missing decision.json and no complete final JSON envelope")
    if any(not isinstance(raw, str) for raw in value["files"].values()):
        raise ValueError("text envelope files must be complete UTF-8 strings")
    files = {"decision.json": _json(value["decision"])}
    for name, raw in value["files"].items():
        _member(name)
        if Path(name).name != name:
            raise ValueError("text envelope design files must be flat")
        files["design/" + name] = raw.encode("utf-8")
    return files


def _evidence(items, inputs):
    if not isinstance(items, list) or len(items) > 40:
        raise ValueError("evidence must be a bounded list")
    checked = []
    for item in items:
        if not isinstance(item, dict) or set(item) != {"source", "start", "end", "claim"}:
            raise ValueError("evidence needs source/start/end/claim")
        source, start, end = item["source"], item["start"], item["end"]
        if not isinstance(source, str) or source not in inputs:
            raise ValueError("evidence source is outside the frozen role view")
        lines = inputs[source].decode("utf-8").splitlines()
        if (type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines)
                or end - start >= 200 or not isinstance(item["claim"], str) or not item["claim"].strip()):
            raise ValueError("evidence needs an existing bounded line range and claim")
        checked.append({**item, "sha256": _digest(inputs[source])})
    return tuple(checked)


def _decision(raw, inputs):
    if len(raw) > MAX_DECISION_BYTES:
        raise CapacityError("decision.json exceeds 128 KiB")
    value = _loads(raw)
    fields = {"status", "rationale", "hypothesis_key", "changed_execution_decisions", "changes", "evidence", "next_actions"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("decision.json must contain exactly " + ", ".join(sorted(fields)))
    if not isinstance(value["status"], str) or value["status"] not in {"replace", "keep", "stop"}:
        raise ValueError("decision status must be replace, keep or stop")
    for key in ("rationale", "hypothesis_key"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError("decision requires " + key)
    for key in ("next_actions", "changed_execution_decisions"):
        if (not isinstance(value[key], list) or len(value[key]) > 40
                or any(not isinstance(item, str) or not item.strip() for item in value[key])):
            raise ValueError(key + " must be a bounded list of nonempty strings")
    if not value["next_actions"]:
        raise ValueError("decision needs a concrete next action")
    changes = value["changes"]
    if not isinstance(changes, list) or len(changes) > 100:
        raise ValueError("changes must be a bounded structured list")
    for item in changes:
        if (not isinstance(item, dict) or set(item) != {"path", "before", "after", "reason"}
                or any(not isinstance(text, str) or not text.strip() for text in item.values())):
            raise ValueError("each change needs nonempty path/before/after/reason strings")
    if value["status"] == "replace" and (not changes or not value["changed_execution_decisions"]):
        raise ValueError("replace needs structured changes and execution decisions")
    value["evidence"] = _evidence(value["evidence"], inputs)
    return value


def _execution_view(checked):
    def strip(value):
        if isinstance(value, dict):
            return {key: strip(child) for key, child in value.items()
                    if key not in {"Thesis", "Describe", "describe", "meaning"}}
        if isinstance(value, (list, tuple)):
            return [strip(child) for child in value]
        return value
    return json_material({name: {"DESIGN": strip(node["DESIGN"]), "when": node.get("when")}
                          for name, node in checked.resolved.effective_task.items()})


class Planner:
    def __init__(self, agent, materials_snapshot: ResourcesSnapshot, *,
                 config: PlannerConfig | None = None, observe_result: Callable | None = None):
        self.agent, self.materials_snapshot = agent, materials_snapshot
        self.config, self.observe_result = config or PlannerConfig(), observe_result

    def design(self, request: PlannerRequest) -> PlannerResult:
        if request.current_design is not None:
            raise ValueError("initial design cannot claim an existing design")
        return self._write("design", request)

    def optimize(self, request: PlannerRequest) -> PlannerResult:
        if request.current_design is None:
            raise ValueError("optimization requires a committed current design")
        current_policy = read_design(request.current_design).checked.policy
        policy = CheckPolicy(skip_hardware=current_policy.skip_hardware, style=self.config.check_policy.style)
        if policy != self.config.check_policy:
            return Planner(self.agent, self.materials_snapshot,
                config=replace(self.config, check_policy=policy), observe_result=self.observe_result)._write("optimize", request)
        return self._write("optimize", request)

    def _inputs(self, request):
        if not all(isinstance(value, str) and value.strip() for value in (
                request.prepared_id, request.operation_id, request.public_task, request.target_hardware)):
            raise ValueError("writer requires prepared/operation identities and frozen public Task/hardware")
        if type(request.version) is not int or request.version < 1 or type(request.require_new) is not bool:
            raise ValueError("invalid design version or require_new")
        feedback = request.feedback
        if isinstance(feedback, Mapping):
            feedback = (dict(feedback),)
        raw_feedback = _json(feedback)
        if len(raw_feedback) > MAX_FEEDBACK_BYTES:
            raise ValueError("Planner feedback exceeds 32 KiB; the controller must bound it")
        inputs = role_material_view(self.materials_snapshot)
        inputs.update({"task/public.md": request.public_task.encode(),
                       "task/hardware.md": request.target_hardware.encode(),
                       "task/feedback.json": raw_feedback,
                       "task/request.json": _json({"prepared_id": request.prepared_id,
                           "operation_id": request.operation_id, "version": request.version,
                           "require_new": request.require_new})})
        # Only the controller's retained candidate sources are projected. No
        # private evaluator directory or host-relative reference grants access.
        sources = {}
        if len(request.source_files) > 256 or sum(len(raw) for raw in request.source_files.values()) > 16 * 1024 * 1024:
            raise CapacityError("Planner candidate source view exceeds 256 files or 16 MiB")
        for name, raw in sorted(request.source_files.items()):
            _member(name)
            if not name.startswith("sources/") or not isinstance(raw, bytes):
                raise ValueError("Planner source files require frozen sources/ byte entries")
            raw.decode("utf-8")
            sources[name] = {"sha256": _digest(raw), "size_bytes": len(raw)}
            inputs[name] = raw
        inputs["task/sources.json"] = _json(sources)
        if request.current_design:
            current = read_design(request.current_design)
            if current.manifest["prepared_id"] != request.prepared_id or request.version <= current.manifest["version"]:
                raise ValueError("optimization must retain prepared identity and advance version")
            inputs.update({"current-design/" + name: raw for name, raw in current.checked.files.items()})
        return inputs

    def _intent(self, directory, mode, inputs, extra):
        directory = _directory(directory)
        intent = {"format": "aka.schema.role-operation.v1", "mode": mode,
                  "inputs": {name: _digest(raw) for name, raw in sorted(inputs.items())},
                  "config": asdict(self.config), "agent_identity": self.agent.configuration_identity, **extra}
        path = directory / "intent.json"
        if path.exists():
            if _load(path) != intent:
                raise ValueError("operation replay conflicts with frozen role inputs or configuration")
        elif directory.exists() and any(directory.iterdir()):
            raise ValueError("role operation directory has no recoverable intent")
        else:
            durable_write_json(path, intent)
        return directory

    def _call(self, directory, inputs, prompt, operation_id, *, read_only_design=False):
        """A started call without a receipt is unknown and is never relaunched."""
        receipt_path = directory / "native/receipt.json"
        call_id = operation_id + ":attempt:" + directory.name
        expected = {"operation_id": operation_id, "call_id": call_id}
        if (directory / "started.json").exists() and _load(directory / "started.json") != expected:
            raise ValueError("retained role call identity differs from this attempt")
        replay = receipt_path.exists()
        if not replay:
            if (directory / "started.json").exists():
                return None
            durable_write_text(directory / "prompt.md", prompt, mode=0o444)
            durable_write_json(directory / "started.json", expected)
            invoke_role(self.agent, workspace=directory / "workspace",
                        observations_directory=directory / "native", inputs=inputs,
                        prompt=prompt, config=self.config, operation_id=call_id,
                        observe_result=self.observe_result, read_only_design=read_only_design)
        receipt = _load(receipt_path)
        if replay and self.observe_result is not None:
            # The durable call id, not the provider's possibly reused session
            # id, identifies one invocation. Replay observes this same receipt.
            native = NativeInvocation(**{**receipt,
                "stdout": read_source_bytes(directory / "native/stdout.txt").decode(),
                "stderr": read_source_bytes(directory / "native/stderr.txt").decode(),
                "terminal_usage": TokenUsage(**receipt["terminal_usage"]),
                "capabilities": AgentRuntimeCapabilities(**receipt["capabilities"]),
                "events": tuple(NormalizedAgentEvent(**{**event,
                    "usage": TokenUsage(**event["usage"]) if event["usage"] else None})
                    for event in receipt["events"]),
                "observation_errors": tuple(receipt["observation_errors"])})
            self.observe_result(call_id, native)
        return receipt

    def retained_role(self, *, directory, inputs, prompt, operation_id,
                      read_only_design=False, config=None):
        """One persisted worker/reviewer invocation using the native writer facilities.

        The extraction stage owner validates permissions and approves output.
        Returning raw bytes is never an approval. Unknown calls retain their
        original owner and are not silently relaunched.
        """
        runner = self if config is None else Planner(self.agent, self.materials_snapshot,
            config=config, observe_result=self.observe_result)
        directory = runner._intent(directory, "extract-role", inputs,
            {"operation_id": operation_id, "prompt_sha256": _digest(prompt.encode()),
             "read_only_design": read_only_design})
        attempt = directory / "attempts/1"
        receipt = runner._call(attempt, inputs, prompt, operation_id,
                               read_only_design=read_only_design)
        if receipt is None or receipt["timed_out"] or receipt["exit_status"] != 0:
            return receipt, {}, ()
        files, problems = runner._collect(attempt)
        return receipt, files, tuple(problems)

    def _collect(self, attempt):
        saved = attempt / "collected.json"
        if saved.exists():
            manifest = _load(saved)
            files = {name: read_source_bytes(attempt / "raw" / _member(name)) for name in manifest["files"]}
            if {name: _digest(raw) for name, raw in files.items()} != manifest["files"]:
                raise ValueError("retained writer output changed")
            return files, manifest["problems"]
        files, inventory, problems = _capture(attempt / "workspace/output")
        if not files and not problems and set(inventory) <= {"design"}:
            files = _text_envelope(read_source_bytes(attempt / "native/stdout.txt").decode("utf-8"))
            inventory = sorted(files)
        _write_files(attempt / "raw", files)
        durable_write_json(saved, {"files": {name: _digest(raw) for name, raw in files.items()},
                                  "inventory": inventory, "problems": problems})
        return files, problems

    def _write(self, mode, request):
        inputs = self._inputs(request)
        prompt = (read_source_bytes(PROMPTS / "common.md") + b"\n\n" +
                  read_source_bytes(PROMPTS / (mode + ".md"))).decode()
        directory = self._intent(request.directory, mode, inputs,
            {"operation_id": request.operation_id, "prompt_sha256": _digest(prompt.encode())})
        if (directory / "result.json").exists():
            return PlannerResult.from_dict(_load(directory / "result.json"))
        diagnostics = []
        for attempt_number in range(self.config.max_repairs + 1):
            attempt = directory / "attempts" / str(attempt_number + 1)
            attempt_inputs = dict(inputs)
            if diagnostics:
                attempt_inputs["task/repair-diagnostics.json"] = _json(diagnostics)
                previous = directory / "attempts" / str(attempt_number) / "raw"
                if previous.exists():
                    for path in previous.rglob("*"):
                        if path.is_file():
                            attempt_inputs["previous-output/" + path.relative_to(previous).as_posix()] = read_source_bytes(path)
                call_prompt = prompt + "\n\nThis is a bounded repair call. Read task/repair-diagnostics.json and previous-output/ under inputs. Write a fresh complete output resolving those concrete diagnostics."
            else:
                call_prompt = prompt
            receipt = self._call(attempt, attempt_inputs, call_prompt, request.operation_id)
            if receipt is None:
                return PlannerResult("failed", None, request.operation_id, directory,
                    (*diagnostics, _notice("invocation.unknown", "A started native call has no retained receipt; settle its owner before a new operation.")),
                    calls=attempt_number + 1)
            if receipt["timed_out"] or receipt["exit_status"] != 0:
                return self._save(directory, PlannerResult("failed", None, request.operation_id, directory,
                    (*diagnostics, _notice("invocation.failed", f"Native role exit={receipt['exit_status']}, timed_out={receipt['timed_out']}; output is not accepted.")),
                    calls=attempt_number + 1))
            try:
                files, problems = self._collect(attempt)
                if problems:
                    raise ValueError("; ".join(problems))
                if "decision.json" not in files or any(name != "decision.json" and not name.startswith("design/") for name in files):
                    raise ValueError("output requires decision.json and only the complete design directory")
                decision = _decision(files["decision.json"], attempt_inputs)
                design_files = {name.removeprefix("design/"): raw for name, raw in files.items() if name.startswith("design/")}
                status, ref = decision["status"], None
                if mode == "design" and status != "replace":
                    raise ValueError("initial design must provide a complete replacement")
                if request.require_new and status == "keep":
                    raise ValueError("require_new forbids keep; propose a new execution decision or a reasoned stop")
                if status != "replace" and design_files:
                    raise ValueError("keep/stop must not include unpublished design files")
                if status == "replace":
                    checked = self._check(attempt, design_files)
                    diagnostics.extend(asdict(item) for item in checked.diagnostics)
                    if request.current_design:
                        previous = read_design(request.current_design)
                        if checked.resolved.raw_task["root"]["CONTRACT"] != previous.checked.resolved.raw_task["root"]["CONTRACT"]:
                            raise ValueError("optimization cannot change the frozen Contract")
                        if _execution_view(checked) == _execution_view(previous.checked):
                            if request.require_new:
                                raise ValueError("require_new needs a changed mathematical, scheduling or pipeline decision")
                            status = "keep"
                            diagnostics.append(_notice("design.equivalent", "Replacement has no changed execution decision; retained the existing design."))
                    if status == "replace":
                        # Pre-manifest interrupted publications are kept; rebuild
                        # only from the already retained checked bytes.
                        publication = directory / "published"
                        index = 0
                        while publication.exists() and any(publication.iterdir()) and not (publication / "manifest.json").exists():
                            index += 1
                            publication = directory / ("published-recovery-" + str(index))
                        try:
                            ref = publish_design(publication, checked.files, version=request.version,
                                prepared_id=request.prepared_id, operation_id=request.operation_id,
                                materials_snapshot=self.materials_snapshot, parent=request.current_design,
                                mode=mode, policy=self.config.check_policy)
                            read_design(ref)
                        except _DATA_ERRORS as error:
                            raise RuntimeError("Design publication interrupted; resume this operation from its retained candidate") from error
                result = PlannerResult(status, ref, request.operation_id, directory,
                    tuple(diagnostics), decision["rationale"], tuple(decision["changes"]),
                    tuple(decision["evidence"]), tuple(decision["next_actions"]), decision["hypothesis_key"],
                    tuple(decision["changed_execution_decisions"]), attempt_number + 1)
                return self._save(directory, result)
            except (ValueError, UnicodeError, SchemaError, CapacityError, ReadError) as error:
                diagnostics.extend(_diagnostics(error))
                durable_write_json(attempt / "diagnostics.json", diagnostics)
        return self._save(directory, PlannerResult("failed", None, request.operation_id, directory,
                                                  tuple(diagnostics), calls=self.config.max_repairs + 1))

    def _check(self, attempt, files):
        try:
            checked = check_design(files, policy=self.config.check_policy)
        except (SchemaError, ReadError) as original:
            if not self.config.normalize:
                raise
            try:
                plan = normalize_design(files, skip_hardware=self.config.check_policy.skip_hardware)
                candidate = {**files, **{name + ".yaml": text.encode() for name, text in plan.candidates.items()}}
                checked = check_design(candidate, policy=self.config.check_policy)
            except _DATA_ERRORS:
                raise original
            patches = {name: "".join(difflib.unified_diff(files[name].decode().splitlines(True),
                        raw.decode().splitlines(True), fromfile="raw/" + name, tofile="normalized/" + name))
                       for name, raw in candidate.items() if raw != files[name]}
            durable_write_json(attempt / "normalization.json", {
                "source_digest": closure_digest(files), "normalized_digest": closure_digest(candidate),
                "corrections": [asdict(item) for item in plan.corrections], "diffs": patches})
        checked_dir = attempt / "checked"
        marker = attempt / "checked.json"
        if marker.exists():
            if _load(marker) != {"closure_digest": closure_digest(checked.files)} or collect_design(checked_dir) != checked.files:
                raise ValueError("retained checked candidate differs from the complete reread")
        else:
            # Derived bytes are reconstructible from raw; a partial write is
            # completed locally without asking the author to create another draft.
            _write_files(checked_dir, checked.files)
        reread = check_design(checked_dir, policy=self.config.check_policy)
        durable_write_json(marker, {"closure_digest": closure_digest(reread.files)})
        return reread

    @staticmethod
    def _save(directory, result):
        durable_write_json(directory / "result.json", result.as_dict())
        return result

    def qualify(self, *, design_ref: DesignRef, prepared_id: str, operation_id: str,
                directory: Path, source: RevisionContent, source_identity: str,
                evidence: dict) -> dict:
        """Independent model judgment of sealed-source fidelity, not a proof.

        The native owner supplies the actual sealed RevisionContent and its
        source_identity. Host-derived obligations and exact source line ranges
        prevent an empty or self-approved report from qualifying a candidate.
        Correctness, measurement and promotion remain separate native gates.
        """
        artifact = read_design(design_ref)
        if (artifact.manifest["prepared_id"] != prepared_id
                or not isinstance(source_identity, str) or not source_identity
                or not isinstance(operation_id, str) or not operation_id):
            raise ValueError("qualification requires bound design, Task, operation and sealed source")
        if not isinstance(source, RevisionContent):
            raise TypeError("qualification source must come from the native RevisionContent owner")
        inputs = {"current-design/" + name: raw for name, raw in artifact.checked.files.items()}
        source_files = {}
        for name, raw in source.files.items():
            _member(name)
            if raw is not None:
                raw.decode("utf-8")
                source_files[name] = raw
                inputs["candidate/" + name] = raw
        if not source_files:
            raise ValueError("sealed source is empty")
        evidence = dict(evidence)
        public_task = evidence.pop("public_task", None)
        target_hardware = evidence.pop("target_hardware", None)
        if not all(isinstance(value, str) and value.strip() for value in (public_task, target_hardware)):
            raise ValueError("qualification requires the native frozen public Task and target hardware")
        inputs["task/public.md"] = public_task.encode("utf-8")
        inputs["task/hardware.md"] = target_hardware.encode("utf-8")
        seeds = evidence.pop("seed_files", {})
        if not isinstance(seeds, Mapping):
            raise TypeError("seed_files must be the controller's frozen file mapping")
        normalized_seeds = {}
        for name, raw in seeds.items():
            _member(name)
            if raw is not None:
                raw = raw.encode() if isinstance(raw, str) else raw
                if not isinstance(raw, bytes):
                    raise TypeError("seed source must contain bytes")
                raw.decode("utf-8")
                normalized_seeds[name] = raw
                inputs["initial-seed/" + name] = raw
        seeds = normalized_seeds
        raw_evidence = _json(evidence)
        if len(raw_evidence) > MAX_FEEDBACK_BYTES:
            raise ValueError("qualification feedback exceeds 32 KiB")
        obligations = ["root.CONTRACT.Algorithm_requirement", "root.CONTRACT.Hardware_specification", "dispatch"]
        obligations += [name + ".DESIGN." + region for name in sorted(artifact.checked.resolved.raw_task)
                        for region in ("Math_reshape", "Work_schedule", "Tile_pipeline")]
        binding = {"design_identity": design_ref.identity, "prepared_id": prepared_id,
                   "source_identity": source_identity,
                   "public_task_sha256": _digest(inputs["task/public.md"]),
                   "target_hardware_sha256": _digest(inputs["task/hardware.md"]),
                   "sealed_files": {name: _digest(raw) for name, raw in sorted(source_files.items())},
                   "obligations": obligations, "seed_identity": evidence.get("seed_identity")}
        inputs.update({"review/binding.json": _json(binding), "review/evidence.json": raw_evidence})
        prompt = read_source_bytes(PROMPTS / "reviewer.md").decode()
        directory = self._intent(directory, "qualify", inputs,
            {"operation_id": operation_id, "prompt_sha256": _digest(prompt.encode())})
        if (directory / "result.json").exists():
            return _load(directory / "result.json")
        result = {"qualified": False, "verdict": "unknown", "substantive_change": False,
                  "basis": "independent model judgment; not a formal proof",
                  "binding": binding, "operation_id": operation_id, "calls": 0,
                  "directory": str(directory), "diagnostics": [], "evidence": []}
        for attempt_number in range(1, self.config.max_repairs + 2):
            attempt = directory / "attempts" / str(attempt_number)
            attempt_inputs, call_prompt = dict(inputs), prompt
            if result["diagnostics"]:
                attempt_inputs["task/repair-diagnostics.json"] = _json(result["diagnostics"])
                previous = directory / "attempts" / str(attempt_number - 1) / "raw/decision.json"
                if previous.exists():
                    raw = read_source_bytes(previous)
                    if len(raw) <= MAX_DECISION_BYTES:
                        attempt_inputs["previous-output/decision.json"] = raw
                call_prompt += ("\n\nThis is a bounded output-format correction. Read inputs/task/repair-diagnostics.json "
                    "and inputs/previous-output/decision.json when present. Write a fresh complete decision resolving "
                    "those protocol diagnostics for the same frozen Task, hardware, design, candidate and initial seed. "
                    "Preserve substantive judgments: real deviations remain fail and unresolved questions remain unknown. "
                    "A diagnostic is not a request for a passing verdict. Do not change source or design.")
            receipt = self._call(attempt, attempt_inputs, call_prompt, operation_id, read_only_design=True)
            result["calls"] = attempt_number
            if receipt is None:
                result["diagnostics"].append(_notice("invocation.unknown",
                    "Started reviewer call has no retained receipt; settle its owner before retrying."))
                return result
            if receipt["timed_out"] or receipt["exit_status"] != 0:
                result["diagnostics"].append(_notice("invocation.failed", "Reviewer did not complete successfully"))
                break
            # A damaged retained artifact is a host recovery problem, not a
            # reason to buy another review. Fresh output still needs parsing.
            collected = (attempt / "collected.json").exists()
            if collected:
                files, problems = self._collect(attempt)
            try:
                # Reviews write a decision object, without any candidate source
                # or design edits. Use the same complete raw retention protocol.
                if not collected:
                    files, problems = self._collect(attempt)
                if problems or set(files) != {"decision.json"}:
                    raise ValueError("review must return only decision.json")
                if len(files["decision.json"]) > MAX_DECISION_BYTES:
                    raise CapacityError("review decision exceeds 128 KiB")
                value = _loads(files["decision.json"])
                required = {"verdict", "design_identity", "source_identity", "obligations", "rationale",
                            "substantive_change", "change_evidence"}
                if not isinstance(value, dict) or set(value) != required:
                    raise ValueError("review decision has missing or unexpected fields")
                if (value["design_identity"] != design_ref.identity or value["source_identity"] != source_identity
                        or not isinstance(value["verdict"], str) or value["verdict"] not in {"pass", "fail", "unknown"}
                        or not isinstance(value["rationale"], str) or not value["rationale"].strip()
                        or type(value["substantive_change"]) is not bool):
                    raise ValueError("review decision has an invalid binding or verdict")
                rows = value["obligations"]
                if (not isinstance(rows, list) or len(rows) != len(obligations)
                        or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in rows)
                        or sorted(row.get("id", "") for row in rows) != sorted(obligations)):
                    raise ValueError("review must cover each host-defined obligation exactly once")
                evidence_rows = []
                for row in rows:
                    if (set(row) != {"id", "verdict", "reason", "evidence"}
                            or not isinstance(row["verdict"], str) or row["verdict"] not in {"pass", "fail", "unknown"}
                            or not isinstance(row["reason"], str) or not row["reason"].strip()):
                        raise ValueError("review obligation needs verdict, reason and source evidence")
                    references = _evidence(row["evidence"], inputs)
                    if row["verdict"] == "pass" and not any(item["source"].startswith("candidate/") for item in references):
                        raise ValueError("passing obligation lacks an actual sealed-source line citation")
                    evidence_rows.append({**row, "evidence": references})
                changes = _evidence(value["change_evidence"], inputs)
                substantive = value["substantive_change"]
                if substantive and (not seeds or not any(item["source"].startswith("candidate/") for item in changes)
                                    or not any(item["source"].startswith("initial-seed/") for item in changes)):
                    raise ValueError("substantive change needs candidate and fixed initial-seed evidence")
                if all(source_files.get(name) == raw for name, raw in seeds.items() if raw is not None) and set(source_files) == {name for name, raw in seeds.items() if raw is not None}:
                    substantive = False
                result.update(verdict=value["verdict"], rationale=value["rationale"],
                    qualified=value["verdict"] == "pass" and all(row["verdict"] == "pass" for row in rows),
                    substantive_change=substantive, evidence=evidence_rows, change_evidence=changes)
                if value["verdict"] == "pass" and not result["qualified"]:
                    result["verdict"] = "unknown" if any(row["verdict"] == "unknown" for row in rows) else "fail"
                # A well-formed fail/unknown is final; only output protocol
                # diagnostics authorize a bounded, separately admitted call.
                break
            except (ValueError, UnicodeError, CapacityError, ReadError) as error:
                diagnostics = list(_diagnostics(error))
                path = attempt / "diagnostics.json"
                if path.exists():
                    if _load(path) != diagnostics:
                        raise ValueError("retained review diagnostics changed")
                else:
                    durable_write_json(path, diagnostics)
                result["diagnostics"].extend(diagnostics)
        durable_write_json(directory / "result.json", result)
        return result


def coder_context(design_ref: DesignRef, prepared_id: str, episode: int, seed_identity: str,
                  *, next_actions=(), comparison_target=None) -> str:
    artifact = read_design(design_ref)
    if artifact.manifest["prepared_id"] != prepared_id:
        raise ValueError("Coder design belongs to another prepared Task")
    if type(episode) is not int or episode < 1 or not isinstance(seed_identity, str) or not seed_identity:
        raise ValueError("Coder context requires Episode and source seed identity")
    binding = {"design": design_ref.as_dict(), "prepared_id": prepared_id,
               "episode": episode, "seed_identity": seed_identity,
               "closure_digest": artifact.manifest["closure_digest"],
               "planner_next_actions": list(next_actions)}
    comparison = ""
    if comparison_target is not None:
        from aka.workflow.light.measurements import KERNEL_ID
        if not isinstance(comparison_target, str) or KERNEL_ID.fullmatch(comparison_target) is None:
            raise ValueError("Design comparison requires a retained native kernel ID")
        binding["design_comparison_target"] = comparison_target
        baseline = "scratch/schema-baseline-" + comparison_target[len("kernel-"):]
        comparison = ("\nFor a comparison against this design's retained best source, use the native tools:\n```sh\n"
            f"python3 tools/sandbox.py --kind kernel-read --kernel-id {comparison_target} --output-path {baseline}\n"
            f"python3 tools/sandbox.py --kind run --mode full --no-sync --baseline-path {baseline}\n"
            "```\nThe baseline may be a file or a source directory. Follow the separate native acceptance "
            "request for promotion against the champion.\n")
    # The native owner supplies its immutable filesystem mount. Inline the full
    # frozen design as well so the context never depends on a mutable checkout.
    return (read_source_bytes(PROMPTS / "coder.md").decode() + "\n\nBinding:\n" + _json(binding).decode()
            + comparison
            + "\n".join("\nFrozen " + name + ":\n```\n" + raw.decode() + "\n```\n"
                         for name, raw in sorted(artifact.checked.files.items())))
