"""Trusted full-Task evaluation through AKA Evaluator/Executor contracts.

KS owns staging, execution, comparison, timing and original receipt validation.
This adapter owns only native/public source mapping and native result boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import time

from aka.contracts.evaluation import (EvaluationFailure, MeasurementArtifact, PreparedEvaluation,
                                    StepEvaluation, VerificationResult)
from aka.contracts.execution import (BackendCapabilities, CommandSpec, ExecutionFacts,
                                    JobResult, ResourceIdentity, UnsupportedExecution)
from kernelschema.artifacts import Bundle, Publication, durable_mkdir, durable_write, import_bundle, read_regular
from kernelschema.contracts import ProtocolError, digest, jcs, loads, sha256
from kernelschema.evaluation import stage
from kernelschema.evaluation.task import verify_stage

from .source import freeze_native_source
from .task import evaluation_bundle

OPERATION = "schema_task"


def _freeze(value):
    return loads(jcs(value))


def _put_once(path, data):
    if path.exists():
        if read_regular(path.parent, path.name)[0] != data:
            raise ProtocolError("original Schema operation binding changed: " + path.name)
    else:
        durable_write(path, data)


@dataclass(frozen=True)
class EvaluationOperation:
    """One owner-admitted operation; A5 persists/reserves BEFORE factory use."""
    operation_id: str
    submission_key: str
    owner: str
    reservation: dict
    expected_source: dict | None = None
    mode: str = "schema_task"

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.operation_id):
            raise ProtocolError("safe stable operation_id required")
        if not self.submission_key or not self.owner or self.reservation.get("owner") != self.owner:
            raise ProtocolError("matching owner/reservation/stable key required")
        ms = self.reservation.get("active_ms")
        if self.reservation.get("evaluator_jobs") != 1 or type(ms) is not int or not 0 < ms <= 1200000:
            raise ProtocolError("one admitted evaluator job and positive <=20min limit required")
        object.__setattr__(self, "reservation", _freeze(self.reservation))
        if self.mode not in {"v0", "framework", "schema_task"}:
            raise ProtocolError("operation mode must select full Task evaluation")


class LocalChannel:
    """The existing KS CPU executor, with complete timing enabled."""
    def __init__(self, executor):
        self.executor = executor

    def identity(self):
        return self.executor.identity()

    def prepare(self, staged, operation):
        self.executor.prepare(staged, include_timing=True)

    def submit(self, staged, operation):
        return self.executor.submit(staged, **operation)

    def worker(self, facts):
        return facts.get("worker")


class AgateChannel:
    """Inject an existing KS AgateDevExecutor and actual prior probe capture.

    This does not discover auth, install a worker, probe or confer qualification.
    A5 must admit the channel from its actual service/profile evidence first.
    """
    def __init__(self, executor, *, profile, environment, frozen_inputs: bytes):
        self.executor = executor
        self.profile, self.environment = _freeze(profile), _freeze(environment)
        self.frozen_inputs = frozen_inputs

    def identity(self):
        return dict(profile=self.profile, environment=self.environment,
                    frozen_inputs_sha256=sha256(self.frozen_inputs),
                    transport=self.executor.transport.identity())

    def prepare(self, staged, operation):
        self.executor.prepare(staged, **operation, profile=self.profile,
                              environment=self.environment, frozen_inputs=self.frozen_inputs, action="execute")

    def submit(self, staged, operation):
        return self.executor.submit(operation["operation_id"])

    def worker(self, facts):
        return (facts.get("execution") or {}).get("worker")


class SchemaExecution:
    """Narrow AKA Executor transporting one trusted KS evaluation channel.

    All directories are host-private. Unknown/cancelled waiting never resubmits a
    job; the KS original operation remains the only settle/lookup target.
    """
    capabilities = BackendCapabilities(operations=frozenset({OPERATION}), prepared_requests=True, cancellation=True)

    def __init__(self, root: Path, channel):
        self.root = Path(root).absolute()
        self.channel = channel
        self.resource_identity = ResourceIdentity("kernelschema", str(self.root))

    def locator(self, operation_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", operation_id):
            raise ProtocolError("unsafe operation identifier")
        return dict(adapter_root=str(self.root / operation_id),
                    executor=self.channel.executor.locator(operation_id))

    def run(self, command: CommandSpec) -> JobResult:
        self.capabilities.require(command)
        if command.operation != OPERATION or command.submission is None:
            raise UnsupportedExecution("Schema executor accepts only prepared full-Task operations")
        envelope = loads(command.submission)
        operation = envelope["operation"]
        root = self.root / operation["operation_id"]
        if root.is_relative_to(command.workspace.absolute()):
            raise ProtocolError("private evaluation root must be outside Agent workspace")
        if envelope["channel"] != self.channel.identity():
            raise ProtocolError("selected execution channel changed")
        if command.timeout != envelope["timeout"]:
            raise ProtocolError("command timeout differs from admitted operation")
        if read_regular(root, "intent.json")[0] != command.submission:
            raise ProtocolError("command differs from original evaluation intent")
        if command.idempotency_key != operation["submission_key"]:
            raise ProtocolError("native and KS submission keys differ")
        bundle = Bundle(root / "bundle")
        staged_path = root / "staged.json"
        if staged_path.exists():
            staged = loads(read_regular(root, staged_path.name)[0])
            verify_stage(staged)
        else:
            staged = stage(bundle, envelope["task"], envelope["source_binding"]["source"],
                           root / "stage", scope=envelope["scope"])
            _put_once(staged_path, jcs(staged))
        if command.cancel_event is not None and command.cancel_event.is_set():
            return self._pending("cancelled before submission")
        self.channel.prepare(staged, operation)
        if command.cancel_event is not None and command.cancel_event.is_set():
            return self._pending("cancelled after preparation, before submission")
        facts = self.channel.submit(staged, operation)
        deadline = time.monotonic() + command.timeout
        while facts["state"] != "terminal":
            if (command.cancel_event is not None and command.cancel_event.is_set()) or time.monotonic() >= deadline:
                return self._pending("original operation remains settle_only")
            time.sleep(.05)
            facts = self.channel.executor.query(operation["operation_id"])
        settled = self.channel.executor.settle(operation["operation_id"], Publication(bundle.root))
        result = dict(operation_id=operation["operation_id"], receipt_id=facts["receipt_id"],
                      report=settled.get("report"), source_binding=envelope["source_binding"],
                      channel=envelope["channel"])
        _put_once(root / "result.json", jcs(result))
        return JobResult(0, stdout=jcs(result).decode(), exit_code=0, output_scope="command",
                         transport_status="succeeded", resource=self.resource_identity,
                         facts=ExecutionFacts(disposition="terminal"))

    def _pending(self, reason):
        return JobResult(1, stderr=reason, transport_status="infrastructure",
                         resource=self.resource_identity, facts=ExecutionFacts(disposition="unknown"))

    def read_operation(self, operation_id):
        """Recover a retained terminal without preparing against today's workspace."""
        root = Path(self.locator(operation_id)["adapter_root"])
        envelope = loads(read_regular(root, "intent.json")[0])
        raw = read_regular(root, "result.json")[0]
        return self.consume(envelope, JobResult(0, stdout=raw.decode(), exit_code=0,
                            output_scope="command", transport_status="succeeded"))

    def consume(self, envelope, result: JobResult):
        """Read-only original receipt consumption; never accept a caller's pass."""
        result.require_transport()
        operation_id = envelope["operation"]["operation_id"]
        root = self.root / operation_id
        actual = loads(read_regular(root, "result.json")[0])
        if loads(result.stdout.encode()) != actual:
            raise ProtocolError("native result differs from original Schema receipt")
        if actual["source_binding"] != envelope["source_binding"] or actual["channel"] != envelope["channel"]:
            raise ProtocolError("receipt source/channel differs from prepared operation")
        if envelope["channel"] != self.channel.identity():
            raise ProtocolError("execution channel changed before receipt consumption")
        facts = self.channel.executor.query(operation_id)  # KS verifies loaded bytes/env/input.
        if facts["state"] != "terminal" or facts["receipt_id"] != actual["receipt_id"]:
            raise ProtocolError("original receipt has no matching terminal")
        bundle = Bundle(root / "bundle")
        report = bundle.document(actual["report"], "evaluation") if actual["report"] else None
        worker = self.channel.worker(facts)
        if report is not None:
            absorption = loads(read_regular(Path(self.channel.executor.locator(operation_id)["root"]), "absorption.json")[0])
            if absorption["report"] != actual["report"]:
                raise ProtocolError("report differs from original executor absorption")
            bundle.closure([actual["report"]], subjects=[envelope["task"], envelope["source_binding"]["source"]])
            verify_report_binding(bundle, report, envelope, worker)
        return dict(bundle=bundle, report_ref=actual["report"], report=report, worker=worker,
                    receipt_id=actual["receipt_id"], source_binding=actual["source_binding"],
                    channel=actual["channel"])


def verify_report_binding(bundle, report, envelope, worker):
    """Applicability checks on an ORIGINAL trusted receipt, not generic reuse approval."""
    task = bundle.document(envelope["task"], "task")
    evaluator = bundle.document(task["evaluator"], "evaluator")
    expected = dict(task_sha256=envelope["task"]["sha256"],
                    source_sha256=envelope["source_binding"]["source"]["sha256"],
                    evaluator_sha256=task["evaluator"]["sha256"], inputs_sha256=evaluator["inputs"]["sha256"],
                    timing_sha256=evaluator["timing"]["sha256"], environment_sha256=report["environment"]["sha256"])
    if report["operation_id"] != envelope["operation"]["operation_id"] or report["subject"] != expected:
        raise ProtocolError("evaluation report belongs to different subjects")
    bundle.closure([envelope["task"], envelope["source_binding"]["source"], envelope["scope"]])
    if worker is None:
        if any(report[name]["status"] == "pass" for name in ("build", "correctness", "performance")):
            raise ProtocolError("passing report has no original worker facts")
        return
    if (worker["source"]["sha256"] != expected["source_sha256"] or worker["evaluator"] != task["evaluator"]
            or worker["loaded"]["kernel.py"] != envelope["source_binding"]["kernel_sha256"]):
        raise ProtocolError("actual loaded C/evaluator differs from native source mapping")
    if loads(bundle.read(report["environment"])) != worker["environment"]:
        raise ProtocolError("report environment differs from original execution")
    if not any(ref["sha256"] == worker["frozen_inputs_sha256"] for ref in report["raw_receipts"]):
        raise ProtocolError("report lacks original frozen input/seed bytes")
    if report["samples"] is not None and bundle.read(report["samples"]) != jcs(worker["result"]["timing"]):
        raise ProtocolError("timing samples differ from original full-forward observations")
    result = worker["result"]
    correct, timing = result.get("correctness"), result.get("timing")
    if report["case_ids"] != sorted({call["case_id"] for call in correct["basic_calls"]} if correct else set()):
        raise ProtocolError("report coverage differs from actual evaluator calls")
    if report["build"]["status"] == "pass" and result["build"] != "pass":
        raise ProtocolError("build pass contradicts original execution")
    if report["correctness"]["status"] == "pass" and (
            not correct or not correct["passed"] or worker["environment"].get("mode") != "cuda"):
        raise ProtocolError("correctness pass lacks original CUDA evidence")
    if report["performance"]["status"] == "pass" and (
            not timing or not timing["complete"] or report["samples"] is None
            or worker["environment"].get("mode") != "cuda"):
        raise ProtocolError("performance pass lacks original complete CUDA timing")


def require_applicable(evidence, *, task_bundle, task, source_binding, channel,
                       environment, frozen_inputs_sha256):
    """Check old ORIGINAL E against current explicit execution requirements.

    Pass only evidence returned by SchemaExecution.consume. This never reissues
    a report, changes its operation/hash, or grants Schema/native qualification.
    Environment and RNG bytes cannot be inferred from equal shapes or C alone.
    """
    report, worker = evidence["report"], evidence["worker"]
    if report is None or worker is None:
        raise UnsupportedExecution("original complete Task report is unavailable")
    document = task_bundle.document(task, "task")
    evaluator = task_bundle.document(document["evaluator"], "evaluator")
    expected = dict(task_sha256=task["sha256"], source_sha256=source_binding.source["sha256"],
                    evaluator_sha256=document["evaluator"]["sha256"], inputs_sha256=evaluator["inputs"]["sha256"],
                    timing_sha256=evaluator["timing"]["sha256"], environment_sha256=digest(environment))
    if report["subject"] != expected or evidence["channel"] != channel:
        raise ProtocolError("old evaluation is not applicable to current subjects/channel/environment")
    if (worker["environment"] != environment or worker["frozen_inputs_sha256"] != frozen_inputs_sha256
            or worker["loaded"]["kernel.py"] != source_binding.kernel_sha256
            or evidence["source_binding"]["native_identity"] != source_binding.native_identity):
        raise ProtocolError("old evaluation input/seed/loaded source is not applicable")
    return evidence["report_ref"]


def native_verification(evidence):
    """Project verified Task evidence. PASS is Task success, never ABBA improvement."""
    report, worker = evidence["report"], evidence["worker"]
    failed = report and any(report[name]["status"] == "fail" for name in ("build", "correctness"))
    passed = bool(report and worker and worker["environment"].get("mode") == "cuda"
                  and all(report[name]["status"] == "pass" for name in ("build", "correctness", "performance"))
                  and not report.get("extensions", {}).get("information", {}).get("synthetic"))
    latencies = {}
    if passed:
        bundle = evidence["bundle"]
        evaluator = bundle.document(worker["evaluator"], "evaluator")
        cases = {case["case_id"]: case["coverage_id"] for case in loads(bundle.read(evaluator["inputs"]))["cases"]}
        latencies = {cases[item["case_id"]]: float(item["models"]["A"]["median_ns"]) / 1000
                     for item in worker["result"]["timing"]["summaries"]}
        if not latencies or any(not math.isfinite(value) or value <= 0 for value in latencies.values()):
            raise ProtocolError("qualified timing lacks positive complete-forward medians")
    latency = math.exp(sum(map(math.log, latencies.values())) / len(latencies)) if latencies else None
    failure = None if passed else EvaluationFailure("candidate" if failed else "capability",
        "schema_task_failed" if failed else "schema_task_unqualified",
        "Full Task evidence retained; no CUDA Task pass" if not failed else "Original evaluator rejected a Task postcondition",
        bool(failed))
    artifacts = (MeasurementArtifact("schema-source-binding.json", jcs(evidence["source_binding"])),)
    if report is not None:
        artifacts += (MeasurementArtifact("schema-evaluation.json", evidence["bundle"].read(evidence["report_ref"])),)
    return VerificationResult("PASS" if passed else "FAIL" if failed else "UNAVAILABLE", latency, None, None,
                              failure=failure, artifacts=artifacts, candidate_latency_us_by_shape=latencies)


class SchemaEvaluator:
    def __init__(self, configuration, candidate_workspace, evaluation_input, *, execution, operation):
        self.configuration, self.workspace, self.evaluation_input = configuration, candidate_workspace, evaluation_input
        self.execution, self.operation = execution, operation

    def prepare(self, request):
        if request.mode not in {"v0", "framework", "schema_task"}:
            raise UnsupportedExecution("Schema Task measurement is separate from native ABBA/promotion and reviewer probes")
        bundle, binding = evaluation_bundle(self.evaluation_input)
        if request.expected_shape_ids is not None:
            from kernelschema.evaluation import RowSoftmaxTask
            ids = RowSoftmaxTask(bundle, binding["task"]).evaluation_capabilities()["coverage_ids"]
            if set(request.expected_shape_ids) != set(ids):
                raise UnsupportedExecution("full Task coverage is required")
        operation = self.operation(request) if callable(self.operation) else self.operation
        if not isinstance(operation, EvaluationOperation):
            raise TypeError("operation resolver must return EvaluationOperation")
        mode = operation.mode if request.mode == "schema_task" else request.mode
        root = self.execution.root / operation.operation_id
        if root.is_relative_to(request.workspace.absolute()):
            raise ProtocolError("Schema private operation root must be outside Agent workspace")
        durable_mkdir(root)
        # The same public Task/scope bytes survive candidate changes and receipt recovery.
        import_bundle(bundle, root / "bundle", [binding["task"], binding["scope"]])
        publication = Publication(root / "bundle", digest(operation.operation_id)[7:39])
        candidate = freeze_native_source(publication, request.workspace, self.workspace,
            revision=request.candidate_commit or None, expected=operation.expected_source)
        task = bundle.document(binding["task"], "task")
        if mode == "v0" and candidate.source["sha256"] != task["baseline_source"]["sha256"]:
            raise ProtocolError("V0 must measure Task.baseline_source; starter/partial is C0 only")
        publication.commit([binding["task"], binding["scope"], candidate.source])
        timeout = min(self.configuration.timeout, operation.reservation["active_ms"] // 1000)
        if timeout < 1:
            raise UnsupportedExecution("less than one second remains for Task evaluation")
        submitted = dict(operation_id=operation.operation_id, submission_key=operation.submission_key,
                         owner=operation.owner, reservation={**operation.reservation, "active_ms": timeout * 1000})
        envelope = dict(version=1, task=binding["task"], scope=binding["scope"], source_binding=candidate.as_dict(),
                        operation=submitted, channel=self.execution.channel.identity(), mode=mode,
                        candidate_revision=request.candidate_commit, timeout=timeout,
                        native_locator=self.execution.locator(operation.operation_id))
        raw = jcs(envelope)
        _put_once(root / "intent.json", raw)
        command = CommandSpec(request.workspace, ("kernelschema-evaluate", operation.operation_id),
            timeout=timeout, operation=OPERATION, submission=raw, idempotency_key=operation.submission_key,
            cancel_event=request.cancel_event)
        return PreparedEvaluation((command,), 1, envelope)

    def observe_step(self, prepared, index, result):
        if index != 0:
            raise ProtocolError("full Task has exactly one logical evaluation step")
        self.execution.consume(prepared.context, result)
        return StepEvaluation()  # raw report remains host-private, never mirrored to Agent.

    def interpret(self, prepared, results):
        if len(results) != 1:
            raise ProtocolError("full Task needs its one original result")
        return native_verification(self.execution.consume(prepared.context, results[0]))

    def measurement(self, prepared, result):
        """Original complete Task evidence for A5's separate qualification/ABBA gate."""
        return self.execution.consume(prepared.context, result)


class SchemaEvaluatorFactory:
    """Explicit injection; operation may be a replayable owner resolver(request).

    The resolver selects the original operation on re-prepare/recovery; it must
    not allocate a new budget or key merely because prepare was called again.
    """
    def __init__(self, *, execution: SchemaExecution, operation):
        self.execution, self.operation = execution, operation

    def create(self, configuration, *, candidate_workspace, evaluation_input):
        evaluation_bundle(evaluation_input)
        return SchemaEvaluator(configuration, candidate_workspace, evaluation_input,
                               execution=self.execution, operation=self.operation)

    def create_command(self, configuration, *, evaluation_input):
        from aka.task.candidate_workspace.provider import GitCandidateWorkspace
        from .command import SchemaCommandEvaluator
        return SchemaCommandEvaluator(self.create(configuration, candidate_workspace=GitCandidateWorkspace(),
                                                  evaluation_input=evaluation_input))
