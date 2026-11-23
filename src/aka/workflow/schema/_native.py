"""Schema's narrow adapter to the existing native Environment/Experiment owners.

No model tool is added. Raw Job/route/Task material stays on the host. An
embedding owner may supply a retained Job and a complete, hash-bound Task
review for a new operation; absence of that review never authorizes correction.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Mapping

from aka.contracts.execution_environment import EnvironmentQuery, EnvironmentSnapshot
from aka.core.filesystem import durable_write_text, read_regular_bytes
from .execution_context import (CORRECTABLE, VERSION_TOKEN, build_execution_context,
                                execution_binding, retain_legacy_device_label, sha256)
from .materials import DesignRef, _json
from .run_state import digest
from aka.task.source.inspection import framework_key


@dataclass(frozen=True)
class TaskConstraintReview:
    """Explicit trusted embedding-owner review of the complete frozen public Task.

    ``constraints`` contains all three CORRECTABLE leaves, each with status
    explicit/unspecified/unknown and canonical value or None. This is an owner
    attestation, not an NLP inference or a model-authored permission bit.
    """
    public_task_sha256: str
    constraints: Mapping
    # All three fields explicitly attest a reviewed historical display label;
    # they grant no Device edits and never alter the native execution target.
    legacy_device_parent: DesignRef | None = None
    legacy_device_label: str | None = None
    legacy_device_target: str | None = None

    def inspect(self, public_task):
        if sha256(public_task.encode()) != self.public_task_sha256:
            raise ValueError("Task review does not bind the complete public Task")
        if set(self.constraints) != set(CORRECTABLE):
            raise ValueError("Task review must explicitly cover all correctable leaves")
        return json.loads(json.dumps(self.constraints))

    def bind_device_label(self, context):
        if all(value is None for value in (self.legacy_device_parent, self.legacy_device_label, self.legacy_device_target)):
            return context
        if (not isinstance(self.legacy_device_parent, DesignRef) or not isinstance(self.legacy_device_label, str)
                or not isinstance(self.legacy_device_target, str)
                or context.as_dict()["binding"]["public_task_sha256"] != self.public_task_sha256):
            raise ValueError("legacy Device review requires an exact parent, label, target and Task")
        return retain_legacy_device_label(context, parent=self.legacy_device_parent,
            label=self.legacy_device_label, target_hardware=self.legacy_device_target)


@dataclass(frozen=True)
class RetainedJobEvidence:
    """Original GET receipt plus its exact owner-checked Experiment identity.

    record_sha256 is digest(record.as_dict()), not the experiment ID. The raw
    receipt hash must be supplied from its retained provenance, never computed
    from a model attachment and treated as authority. No fetch occurs here.
    """
    raw: bytes
    raw_sha256: str
    experiment: object
    record_sha256: str


def _device(value):
    # Vendor prefix normalization only; this never derives an architecture.
    return value.removeprefix("NVIDIA ").strip().casefold() if isinstance(value, str) else None


def _route(executor):
    # These are the selected native adapters' existing route constructors. They
    # perform no query. Do not infer a route from os.environ in Schema itself.
    factory = getattr(executor, "_environment_backend", None)
    return factory() if callable(factory) else None


def _version(values, framework, source):
    if not isinstance(values, Mapping):
        return None, "unknown"
    framework = framework_key(framework)
    name = {"cuda": "nvcc", "pytorch": "torch"}.get(framework, framework)
    value = values.get(name)
    if isinstance(value, Mapping):
        if value.get("available") is False or value.get("supported") is False:
            return None, "unknown"
        declared_source = value.get("source")
        if declared_source in {"configured", "declared", "cache", "unknown"}:
            source = declared_source if source == "observed" else source
        value = value.get("version")
    if source == "unknown" or not isinstance(value, str) or len(value) > 4096:
        return None, "unknown"
    if framework == "cuda":
        # SSH retains nvcc's standard multi-line stdout. Export only its exact
        # compiler version, never banner text or an arbitrary embedded token.
        releases = re.findall(r"(?m)^Cuda compilation tools, release ([0-9]{1,4}\.[0-9]{1,4}), "
                              r"V([0-9]{1,4}(?:\.[0-9]{1,4}){2})$", value)
        if len(releases) == 1 and releases[0][1].startswith(releases[0][0] + "."):
            value = releases[0][1]
    if re.fullmatch(VERSION_TOKEN, value) is None:
        return None, "unknown"
    return value, source


def _snapshot_facts(snapshot, framework):
    source = snapshot.source
    observed = snapshot.observed
    if source in {"declared", "unknown"}:
        values = snapshot.declarations.get("environment", {})
        source = "declared" if source == "declared" else "unknown"
    else:
        values = observed
        # Only known native discovery sources constitute observations. Unknown
        # source labels never acquire authority merely by carrying an arch key.
        source = source if source in {"cache", "configured"} else "observed" if source in {
            "observed", "runtime", "probe", "runtime-probe", "ssh-probe"} else "unknown"
    if not isinstance(values, Mapping):
        values = {}
    arch = values.get("arch") if isinstance(values, Mapping) else None
    version, version_source = _version(values.get("frameworks"), framework, source)
    if version is None:
        version, version_source = _version(values.get("toolchain"), framework, source)
    return {"architecture": {"value": arch if arch and source != "unknown" else None,
                "source": source if arch else "unknown"},
            "toolchain": {"framework": framework,
                "value": version, "source": version_source}}


def _validate_snapshot(snapshot, *, target, route):
    from aka.execution.environment_query import safe_endpoint
    if (not isinstance(snapshot, EnvironmentSnapshot) or _device(snapshot.hardware) != _device(target)
            or snapshot.observed.get("gpu") and _device(snapshot.observed["gpu"]) != _device(target)
            or route is None or snapshot.backend != route.backend
            or snapshot.endpoint != safe_endpoint(route.endpoint)):
        raise ValueError("native environment belongs to another target or selected route")


def _record_task(records, record, campaign, prepared):
    from .progress import _full_task
    if (record.ref.run_id != str(campaign.workspace.absolute()) or record.lifecycle != "terminal"
            or not _full_task(records, record, campaign.problem.evaluation, prepared["hardware"])):
        raise ValueError("native execution evidence does not bind the complete prepared Task")


def _job_facts(evidence, *, records, campaign, prepared, route):
    from aka.execution.gateway.requests import operation_request
    from aka.execution.live_eval import WIRE_PATH, digest as wire_digest
    if not isinstance(evidence.raw, bytes) or sha256(evidence.raw) != evidence.raw_sha256:
        raise ValueError("retained Job bytes differ from their original digest")
    record = records.get(evidence.experiment)
    if digest(record.as_dict()) != evidence.record_sha256:
        raise ValueError("retained native record differs from its original digest")
    _record_task(records, record, campaign, prepared)
    body = json.loads(evidence.raw)
    matches = [(attempt, ref) for attempt in record.attempts if attempt.result
        for ref in attempt.result.job_references if ref.identifier == body.get("job_id")]
    if len(matches) != 1 or body.get("status") != "succeeded":
        raise ValueError("retained Job is not the exact completed native owner")
    attempt, ref = matches[0]
    step = next(item for item in record.steps if item.task_id == attempt.task_id)
    _validate_snapshot(attempt.result.environment_snapshot, target=prepared["hardware"], route=route)
    if (attempt.disposition != "terminal" or ref.operation != "run" or not ref.idempotency_key
            or ref.endpoint != route.endpoint or body.get("num_gpus") != record.request.get("num_gpus")):
        raise ValueError("retained Job route or allocation differs from native request")
    documents = {role: [json.loads(records.read_artifact(item)) for item in record.artifacts if item.role == role]
                 for role in ("evaluation_input", "live_eval_association")}
    associations = [item for item in documents["live_eval_association"]
        if item.get("attempt_id") == attempt.attempt_id and item.get("terminal") is True
        and item.get("job_reference", {}).get("identifier") == ref.identifier
        and item.get("request_identity") == ref.request_identity
        and item.get("route_identity") == wire_digest([route.endpoint, WIRE_PATH, "", ""])
        and item.get("namespace_identity") == wire_digest(route.environment.get("AGATE_AK", ""))]
    if len(associations) != 1 or not route.environment.get("AGATE_AK"):
        raise ValueError("retained Job lacks its exact native route/namespace association")
    requests = []
    for original in documents["evaluation_input"]:
        if original.get("kind") != "run" or original.get("schema_version") != 1:
            continue
        request = operation_request(original["request"], "run", ssh=False)
        request["idempotency_key"] = ref.idempotency_key
        request.setdefault("options", {})["timeout_s"] = step.timeout
        if wire_digest(request) == ref.request_identity:
            requests.append(original)
    if len(requests) != 1:
        raise ValueError("retained Job request does not match the original native input")
    original = requests[0]
    framework = framework_key(prepared["native_policy"]["framework"])
    result = body.get("result", {})
    environment = result.get("environment", {})
    summary = body.get("request_digest", {})
    shape = result.get("shape_selection", {})
    if (not isinstance(result.get("dsl"), str) or framework_key(result["dsl"]) != framework
            or _device(environment.get("gpu_name")) != _device(prepared["hardware"])
            or _device(summary.get("gpu")) != _device(prepared["hardware"])
            or summary.get("num_gpus") != record.request.get("num_gpus")
            or set(shape.get("all_shape_ids", ())) != {point.shape_id for point in step.measurement_identity.points}
            or shape.get("partial") is not False):
        raise ValueError("retained Job Task, target or selected framework is inapplicable")
    sources = [records.read_source(item) for item in record.artifacts if item.role == "candidate-source"]
    if len(sources) != 1 or sources[0].files.get("kernel.py") is None:
        raise ValueError("retained Job candidate source owner is unavailable")
    expected = {name: sha256(raw.encode()) for name, raw in original["reference_files"].items()
                if name in {"input.py", "reference.py", "metadata.json", "shapes.json"}}
    expected["candidate.py"] = sha256(sources[0].files["kernel.py"])
    if shape.get("source_hashes") != expected:
        raise ValueError("retained Job source/Task hashes differ from native artifacts")
    arch = environment.get("gpu_arch")
    snapshot = attempt.result.environment_snapshot
    if snapshot.observed.get("arch") not in {None, arch}:
        raise ValueError("retained runtime and native environment architecture conflict")
    version, version_source = _version({"triton": environment.get("triton_version"),
        "torch": environment.get("torch_version")}, framework, "observed")
    snapshot_version = _snapshot_facts(snapshot, framework)["toolchain"]["value"]
    if version is not None and snapshot_version not in {None, version}:
        raise ValueError("retained runtime and native environment toolchain conflict")
    return {"architecture": {"value": arch, "source": "observed" if arch else "unknown"},
            "toolchain": {"framework": framework, "value": version,
                          "source": version_source}}


def _preparation_job(records, record, campaign, prepared, route):
    """Select an original full successful seed receipt, without acquiring facts.

    Older/native adapters lacking an owned Job keep their snapshot fallback.
    A selected receipt still passes the same verifier as explicit host evidence.
    """
    from .progress import measurement_evidence
    owner = getattr(campaign, "candidate_workspace", None)
    if (route is None or getattr(route, "ssh", False) or owner is None
            or not prepared.get("source_identity")
            or not any(item.role == "live_eval_association" for item in record.artifacts)):
        return None
    latest = {attempt.task_id: attempt for attempt in record.attempts}
    for attempt in reversed(tuple(latest.values())):
        result = attempt.result
        if (result is None or attempt.disposition != "terminal" or result.transport_status != "succeeded"
                or not any(ref.operation == "run" and ref.idempotency_key for ref in result.job_references)):
            continue
        raw = result.stdout.encode()
        try:
            body = json.loads(raw)
        except ValueError:
            continue
        if (not isinstance(body, dict) or body.get("status") != "succeeded"
                or not isinstance(body.get("result"), dict)
                or not isinstance(body["result"].get("shape_selection"), dict)
                or body["result"]["shape_selection"].get("partial") is not False):
            continue
        owners = [ref for ref in result.job_references if ref.identifier == body.get("job_id")]
        if len(owners) != 1 or owners[0].operation != "run":
            raise ValueError("native preparation receipt differs from its own attempt's Job")
        measured = measurement_evidence(records, record, owner=owner,
            task=campaign.problem.evaluation, hardware=prepared["hardware"])
        if (measured is None or not measured["performance"] or any(
                item.identity.method != "atrex-bench/eval" for item in measured["measurements"])):
            return None
        if measured["source_identity"] != prepared["source_identity"]:
            raise ValueError("native preparation Job belongs to another prepared source")
        return RetainedJobEvidence(raw, sha256(raw), record.ref, digest(record.as_dict()))
    return None


def execution_context(*, campaign, prepared, executor, directory, task_review=None,
                      retained_job=None, preparation_records=()):
    """Build one new operation's context from native owners, never during replay.

    Prefer an owned full successful preparation Job over its EnvironmentSnapshot;
    validate the original receipt rather than relabelling cached observations.
    If neither exists, public EnvironmentExecutor discovery may run once on the
    selected bound native workflow, retaining private bytes before admission.
    Legacy/custom embeddings without that native owner retain unknown facts.
    """
    if campaign.problem_provider.snapshot(campaign.problem).identity != prepared["identity"]:
        raise ValueError("execution evidence prepared Task identity changed")
    binding = execution_binding(prepared_id=prepared["identity"], public_task=prepared["public_task"],
        target_hardware=prepared["hardware"], native_policy=prepared["native_policy"])
    framework = prepared["native_policy"]["framework"]
    route = _route(executor)
    evidence, projection = None, None
    reader = getattr(campaign, "experiment_record_reader", None)
    if retained_job is not None:
        if not isinstance(retained_job, RetainedJobEvidence) or not callable(reader):
            raise ValueError("retained Job requires the native Experiment reader")
        evidence = (retained_job.raw, retained_job.raw_sha256)
    elif callable(reader) and preparation_records:
        with reader() as records:
            for ref in reversed(preparation_records):
                record = records.get(ref)
                _record_task(records, record, campaign, prepared)
                retained_job = _preparation_job(records, record, campaign, prepared, route)
                if retained_job is not None:
                    evidence = (retained_job.raw, retained_job.raw_sha256)
                    break
                for attempt in reversed(record.attempts):
                    snapshot = attempt.result.environment_snapshot if attempt.result else None
                    if snapshot is None:
                        continue
                    _validate_snapshot(snapshot, target=prepared["hardware"], route=route)
                    raw = _json(snapshot.as_dict())
                    evidence = (raw, sha256(raw))
                    projection = {"binding": binding, **_snapshot_facts(snapshot, framework)}
                    break
                if evidence:
                    break
    native_workflow = getattr(getattr(campaign, "tool_runtime", None), "workflow", None)
    # A new Schema operation may use the gateway's GET-only discovery. SSH
    # observations must already belong to native preparation; Schema does not
    # open an extra remote probe. Unsupported custom routes remain unknown.
    if not evidence and native_workflow is not None and route is not None and not route.ssh:
        path = directory / "native-environment.json"
        if path.exists():
            raw = read_regular_bytes(path, limit=2 * 1024 * 1024)
            snapshot = EnvironmentSnapshot(**json.loads(raw))
        else:
            snapshot = executor.query_environment(EnvironmentQuery(prepared["hardware"], languages=(framework,)))
            _validate_snapshot(snapshot, target=prepared["hardware"], route=route)
            raw = _json(snapshot.as_dict())
            durable_write_text(path, raw.decode(), mode=0o600)
        _validate_snapshot(snapshot, target=prepared["hardware"], route=route)
        evidence = (raw, sha256(raw))
        projection = {"binding": binding, **_snapshot_facts(snapshot, framework)}
    def verify(raw):
        if retained_job is not None:
            with reader() as records:
                facts = _job_facts(retained_job, records=records, campaign=campaign, prepared=prepared, route=route)
            return {"binding": binding, **facts}
        # The snapshot bytes above came from the selected public query or a
        # complete Task-matched native record, never a caller-supplied mapping.
        return projection
    context = build_execution_context(prepared_id=prepared["identity"], public_task=prepared["public_task"],
        target_hardware=prepared["hardware"], native_policy=prepared["native_policy"], evidence=evidence,
        verify_evidence=verify,
        inspect_task=task_review.inspect if isinstance(task_review, TaskConstraintReview) else None)
    return task_review.bind_device_label(context) if isinstance(task_review, TaskConstraintReview) else context
