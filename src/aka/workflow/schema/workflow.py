"""One persistent Schema controller around the original Light Episode engine."""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Mapping

from aka.core.active_time import ActiveBudget, budget_scope
from aka.core.filesystem import read_regular_bytes
from .materials import DesignRef, read_design, snapshot_ref
from .run_state import BudgetDenied, RecoveryRequired, RunLimits, RunState, digest, execution_attempt_identity
from .progress import needs_new_hypothesis
from .qualification_feedback import applies_to, coder_feedback, project, unavailable
from .execution_context import (authorize_contract_correction, restore_execution_context,
                                check_contract_change, validate_contract_correction,
                                validate_design_execution_context, publication_execution_context,
                                check_context_hardware)


WINDOW_EPISODES = 4
FEEDBACK_CHARACTERS = 8000


@dataclass(frozen=True)
class ExistingStart:
    """Explicitly supplied same-target kernel/design pair, verified by preparation.

    evidence_refs are retained provenance supplied by the preparation owner;
    a filename or a claim of SOTA alone is not pairing evidence.
    """
    design: DesignRef
    source_identity: str
    prepared_id: str
    hardware: str
    evidence_refs: tuple[str, ...]

    def as_dict(self):
        return {"design": self.design.as_dict(), "source_identity": self.source_identity,
                "prepared_id": self.prepared_id, "hardware": self.hardware,
                "evidence_refs": list(self.evidence_refs)}


@dataclass(frozen=True)
class SuppliedStart:
    """Frozen public kernel/design declaration, bound only after native V0.

    The hashes describe the full native source, not a reference implementation.
    Evidence remains a user-supplied provenance declaration; native validation
    and each later candidate's independent qualification are still required.
    """
    files: Mapping[str, bytes]
    source_files: Mapping[str, str]
    hardware: str
    evidence: Mapping[str, bytes]
    policy: object

    def as_dict(self):
        import hashlib
        return {"kind": "supplied", "files": {name: hashlib.sha256(raw).hexdigest()
                    for name, raw in sorted(self.files.items())},
                "source_files": dict(self.source_files), "hardware": self.hardware,
                "evidence": {name: hashlib.sha256(raw).hexdigest()
                    for name, raw in sorted(self.evidence.items())},
                "check_policy": self.policy.as_dict()}

    def bind(self, *, campaign, snapshot, source, directory, materials_snapshot):
        import hashlib
        from aka.core.filesystem import durable_write_text
        from .materials import publish_design
        actual = {name: hashlib.sha256(raw).hexdigest() for name, raw in source.files.items()
                  if raw is not None}
        if actual != dict(self.source_files) or self.hardware != campaign.sandbox_hardware:
            raise ValueError("Supplied kernel/Schema does not match the full prepared native source and target hardware")
        if not self.evidence:
            raise ValueError("Supplied kernel/Schema requires explicit pairing provenance")
        inputs = directory / "initial-inputs"
        evidence_refs = []
        for name, raw in sorted(self.evidence.items()):
            path = inputs / name
            if path.exists() and read_regular_bytes(path, limit=8 * 1024 * 1024) != raw:
                raise ValueError("Retained initial pairing provenance changed")
            durable_write_text(path, raw.decode("utf-8"), mode=0o444)
            evidence_refs.append(str(path))
        # A failed manifest publication is retained; select a new host directory
        # for the same frozen input without authoring or invoking a model.
        number = 0
        while True:
            target = directory / ("initial-design" if number == 0 else f"initial-design-{number}")
            if not target.exists() or (target / "manifest.json").exists() or not any(target.iterdir()):
                break
            number += 1
        design = publish_design(target, self.files, version=1, prepared_id=snapshot.identity,
            operation_id="supplied-initial", materials_snapshot=materials_snapshot,
            mode="import", policy=self.policy)
        return ExistingStart(design, campaign.candidate_workspace.source_identity(source),
            snapshot.identity, self.hardware, tuple(evidence_refs))


@dataclass(frozen=True)
class SotaStart:
    """Explicit SOTA source without a supplied design; never inferred from a seed.

    target_gpu is the separately retained Environment declaration/observation;
    target_hardware must also equal the native campaign's frozen hardware.
    """
    descriptor: object
    target_gpu: Mapping
    target_hardware: str

    def as_dict(self):
        return {"descriptor": self.descriptor.as_dict(), "target_gpu": dict(self.target_gpu),
                "target_hardware": self.target_hardware}


def extract(request, *, planner, config=None):
    from .extract_stages import extract as run_extract
    return run_extract(request, planner=planner, config=config)


def _public_text(public, *, operator_files=()):
    files = {}
    # prepare_public is an authoring delta: ordinary exposed-shape Tasks are
    # ready with no added files. The native AgentProblem role is their already
    # approved Task content; evaluator/worker resources never join this view.
    for item in (*operator_files, *public.files):
        content = item.text if item.text is not None else read_regular_bytes(item.source, limit=16 * 1024 * 1024).decode("utf-8")
        files[item.path] = content
    return "\n\n".join(f"## {name}\n{value}" for name, value in sorted(files.items()))


def _record_ref(record):
    return record.ref.as_dict()


def _record_key(record):
    return digest(_record_ref(record))


def _measurement_feedback(record):
    """Positive public projection; raw runs, payloads and error text stay private."""
    result = {"experiment": _record_ref(record), "phase": record.phase,
              "failure": record.failure.code if record.failure else None,
              "source_artifacts": [{"sha256": item.sha256, "size_bytes": item.size_bytes}
                  for item in record.artifacts if item.role == "candidate-source"]}
    evaluation = record.evaluation
    if evaluation is not None:
        result.update({key: getattr(evaluation, key) for key in (
            "gate", "candidate_latency_us", "incumbent_latency_us", "improvement_pct",
            "candidate_performance_score", "incumbent_performance_score")})
        result["evaluation_failed"] = bool(evaluation.error or evaluation.failure)
    # Native command evaluation retains step assessments on the original
    # attempts; its aggregate verification need not duplicate them. Use Light's
    # existing terminal, task/identity-matched projection for that case.
    from aka.workflow.light.measurements import RecordedMeasurements
    measurements = evaluation.measurements if evaluation is not None else ()
    if not measurements and record.attempts:
        measurements = RecordedMeasurements._measurements(record)
    result["measurements"] = [{"method": item.identity.method,
            "identity_digest": item.identity.digest, "completeness": item.completeness,
            "validity": item.validity, "comparability": item.comparability,
            "environment_identity": item.environment_identity,
            "diagnostics": [{"category": entry.category, "code": entry.code}
                            for entry in item.diagnostics[:8]]} for item in measurements[:8]]
    return result


def build_feedback(state, *, qualification_feedback=None):
    """Bounded public prose/identities, never raw private evaluator inputs."""
    data = state.data
    hypothesis = data["hypotheses"].get(data["hypothesis"], {})
    episodes = list(data["episodes"].values())[-WINDOW_EPISODES:]
    if qualification_feedback is None:
        qualification_feedback = unavailable("qualification-receipt-missing")
    if (qualification_feedback.get("status") == "reviewed" and (not data["design"]
            or not applies_to(qualification_feedback, prepared=data["prepared"],
                design_identity=data["design"]["identity"],
                execution_context_sha256=data.get("execution_context", {}).get("execution_context_sha256")))):
        qualification_feedback = unavailable("qualification-binding-mismatch")
    qualification_feedback = {**qualification_feedback, "scope": "historical-reviewed-source"}
    value = {"design": data["design"], "prepared_id": data["prepared"]["identity"],
        "champion": data["champion"], "tested_candidate": data["tested_candidate"],
        "working_draft": data["working_draft"], "stagnant_windows": hypothesis.get("stagnant", 0),
        "milestones": hypothesis.get("milestones", []), "best_legal_measurement": hypothesis.get("best_legal"),
        "uncertain_windows": hypothesis.get("uncertain_windows", 0),
        "design_comparison_target": ("kernel-" + hypothesis["native_frontier"]["source"]["identity"]
                                     if hypothesis.get("native_frontier") else None),
        "preparation_measurements": data.get("preparation_measurements", []),
        "qualification_feedback": qualification_feedback,
        "episodes": [{key: row.get(key) for key in ("episode", "status", "accepted", "violation", "progress", "progress_assessment", "records", "measurements")}
                     for row in episodes],
        "require_new": needs_new_hypothesis(hypothesis),
        "diagnostics": [{"code": row["code"], "message": row["message"][:600]}
                        for row in data["diagnostics"][-3:]]}
    value = deepcopy(value)
    encode = lambda: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    # Preserve valid JSON and the newest actual result before older context.
    # Full retained owner records remain private; bounded summaries are facts,
    # never permission for Planner to follow arbitrary host paths.
    if len(encode()) > FEEDBACK_CHARACTERS:
        value["truncated"] = "Older context omitted by the 8000-character host limit"
        value["diagnostics"] = []
        value["preparation_measurements"] = []
    while len(encode()) > FEEDBACK_CHARACTERS and len(value["episodes"]) > 1:
        value["episodes"].pop(0)
    for row in value["episodes"]:
        while len(encode()) > FEEDBACK_CHARACTERS and len(row.get("measurements") or []) > 1:
            row["measurements"].pop(0)
        if len(encode()) > FEEDBACK_CHARACTERS:
            row["records"] = (row.get("records") or [])[-1:]
            row["violation"] = str(row.get("violation") or "")[:600]
    if len(encode()) > FEEDBACK_CHARACTERS:
        for role in ("champion", "tested_candidate", "working_draft"):
            value[role] = {"identity": data[role]["identity"]} if data[role] else None
        value["best_legal_measurement"] = None
        for row in value["episodes"]:
            for result in row.get("measurements") or []:
                result["measurements"] = [{key: item[key] for key in (
                    "method", "completeness", "validity", "comparability", "environment_identity")}
                    for item in result.get("measurements", [])[-1:]]
    if len(encode()) > FEEDBACK_CHARACTERS:
        # Specific independent feedback takes precedence over repeated native
        # measurement detail. It is already bounded and identity-bearing.
        value["milestones"] = []
        for row in value["episodes"]:
            row["measurements"], row["records"] = [], []
    text = encode()
    if len(text) > FEEDBACK_CHARACTERS:
        raise RecoveryRequired("Public feedback identities exceed the bounded Planner input")
    return text


class SchemaWorkflow:
    """Light's explicit driver; no copied Agent, evaluation or promotion loop.

    The supplied Planner and Executor must borrow the same Agent/execution
    owners used by Campaign. Public CLI assembly is deliberately separate.
    """
    def __init__(self, directory, *, planner, executor, workspace, limits: RunLimits,
                 initial: ExistingStart | SuppliedStart | None = None, stop_event=None, recover_agent=None,
                 candidate_qualification=None, sleep=time.sleep, sota: SotaStart | None = None,
                 task_constraint_review=None, retained_job_evidence=None):
        if initial is not None and sota is not None:
            raise ValueError("supplied kernel/Schema and source-only extraction are distinct, mutually exclusive C inputs")
        if initial is not None and not isinstance(initial, (ExistingStart, SuppliedStart)) or sota is not None and not isinstance(sota, SotaStart):
            raise TypeError("C input must be an explicit ExistingStart, SuppliedStart or SotaStart")
        from ._native import RetainedJobEvidence, TaskConstraintReview
        if task_constraint_review is not None and not isinstance(task_constraint_review, TaskConstraintReview):
            raise TypeError("Task correction authority requires a complete owner TaskConstraintReview")
        if retained_job_evidence is not None and not isinstance(retained_job_evidence, RetainedJobEvidence):
            raise TypeError("Runtime correction authority requires original RetainedJobEvidence")
        self.directory = Path(directory).absolute()
        self.planner, self.executor, self.initial = planner, executor, initial
        self.sota = sota
        self.stop_event, self.recover_agent, self.sleep = stop_event, recover_agent, sleep
        self.candidate_qualification = candidate_qualification
        # Trusted public embedding inputs for NEW operations only. They are not
        # RootIdentity/config fields and cannot alter an existing frozen action.
        self.task_constraint_review = task_constraint_review
        self.retained_job_evidence = retained_job_evidence
        materials = snapshot_ref(planner.materials_snapshot)
        self.state = RunState(self.directory, limits=limits, workspace=workspace,
            identity={"materials": materials, "initial": initial.as_dict() if initial else None,
                      "sota": sota.as_dict() if sota else None})
        self.root_ref = self.state.root_ref
        self.limits = limits
        self._campaign = None
        self._recovery_executor = executor
        self._original_pending = {}
        self._sota_snapshot = None

    def record_planner_usage(self, operation_id, result):
        usage = result.terminal_usage
        receipt = f"agent:{operation_id}:{result.session_id}"
        self.state.record_usage(receipt, usage.total_tokens, usage.measurement)

    def _native_usage(self, result, workspace):
        usage = getattr(result, "terminal_usage", None)
        with self.state.lock:
            choices = [(permit, call) for permit, call in self.state.data["calls"].items()
                if call["kind"] == "agent" and call["state"] == "settled"
                and call["owner"]["workspace"] == str(workspace) and not call.get("native_usage_claimed")]
            if not choices:
                raise RecoveryRequired("Native helper usage has no actual supervised invocation receipt")
            permit, call = choices[-1]
            call["native_usage_claimed"] = True
        self.state.record_usage(f"native-helper:{permit}",
            usage.total_tokens if usage is not None else None,
            usage.measurement if usage is not None else "unavailable")

    def _sync_native_usage(self, campaign):
        """Project original cumulative Coder receipts before another admission."""
        from aka.workflow.light.long_horizon.store import CampaignStore
        native, _ = CampaignStore.checkpoint(campaign.workspace)
        with self.state.lock:
            seen = self.state.data.setdefault("native_coder_usage", {})
            if set(seen) - set(native.usage_receipts):
                raise RecoveryRequired("Original native Coder usage receipts disappeared")
            for receipt, tokens in native.usage_receipts.items():
                previous = seen.get(receipt, 0)
                if tokens < previous:
                    raise RecoveryRequired("Original native Coder usage decreased during recovery")
                if tokens > previous:
                    seen[receipt] = tokens
                    # Native cumulative receipts contain observed counts, not
                    # a claim of complete provider billing. Per-turn unavailable
                    # usage remains unavailable in the original native telemetry.
                    self.state.record_usage(f"native-coder:{receipt}:{tokens}", tokens - previous, "partial")

    def _scopes(self, stack, campaign):
        if not callable(getattr(campaign.agent, "invocation_scope", None)):
            raise ValueError("Selected Agent does not expose supervised invocation admission")
        if not callable(getattr(self.executor, "dispatch_scope", None)):
            raise ValueError("Selected Executor does not expose supervised dispatch admission")
        if getattr(self.planner, "agent", campaign.agent) is not campaign.agent:
            raise ValueError("Planner and Coder must share the supervised Agent owner")
        factory = getattr(campaign, "tool_runtime", None)
        workflow = getattr(factory, "workflow", None)
        if workflow is not None and workflow.executor is not self.executor:
            raise ValueError("Controller Executor differs from the selected native execution owner")
        if getattr(factory, "wiki", False):
            raise ValueError("Schema requires static materials; dynamic Wiki execution is not admitted")
        def admit_agent(**request):
            self._check_original_records(campaign, model=True)
            self._sync_native_usage(campaign)
            return self.state.admit_agent(**request)
        def admit_execution(**request):
            self._check_original_records(campaign, model=False)
            self._sync_native_usage(campaign)
            return self.state.admit_execution(**request)
        stack.enter_context(campaign.agent.invocation_scope(admit=admit_agent,
            observe=self.state.observe_agent))
        stack.enter_context(self.executor.dispatch_scope(admit=admit_execution,
            observe=self.state.observe_execution))
        stack.enter_context(budget_scope(ActiveBudget(self.limits.max_agent_active_seconds,
            used=self.state.data["counts"]["agent_active_seconds"], checkpoint=self.state.checkpoint_active_time)))
        previous = getattr(self.planner, "observe_result", None)
        def observed(operation_id, result):
            self.record_planner_usage(operation_id, result)
            if previous is not None and previous != self.record_planner_usage:
                previous(operation_id, result)
        self.planner.observe_result = observed
        stack.callback(setattr, self.planner, "observe_result", previous)
        prior_usage = getattr(campaign, "native_usage_observer", None)
        campaign.native_usage_observer = self._native_usage
        stack.callback(setattr, campaign, "native_usage_observer", prior_usage)
        self._capture_unresolved(campaign)
        self._sync_native_usage(campaign)

    def _capture_unresolved(self, campaign):
        if self.state.data["prepared"] is None:
            return
        for record in campaign.experiment_records():
            if record.lifecycle != "terminal":
                self._original_pending[_record_key(record)] = {
                    "ref": _record_ref(record), "attempts": [item.attempt_id for item in record.attempts
                        if item.disposition not in {"terminal", "not_submitted"}]}
        self.state.data["unresolved_experiments"] = list(self._original_pending.values())
        self.state.save()

    def _check_original_records(self, campaign, *, model):
        if not self._original_pending:
            return
        records = {_record_key(record): record for record in campaign.experiment_records()}
        changed = False
        for key, pending in list(self._original_pending.items()):
            record = records.get(key)
            if record is None:
                raise RecoveryRequired("An original unresolved Experiment record disappeared")
            if record.lifecycle == "terminal":
                self._original_pending.pop(key)
                changed = True
                continue
            attempts = {item.attempt_id: item for item in record.attempts}
            # Original native recovery may query and complete the existing
            # request, including an explicitly missing step. It cannot start a
            # new model, or dispatch over an unresolved old attempt.
            if model or not pending["attempts"] or any(attempts.get(item) is None
                    or attempts[item].disposition not in {"terminal", "not_submitted"} for item in pending["attempts"]):
                raise RecoveryRequired("Original Experiment must be recovered before new work: " + str(pending["ref"]))
        if changed:
            self.state.data["unresolved_experiments"] = list(self._original_pending.values())
            self.state.save()

    def _bind_recovery(self, campaign):
        runtime = getattr(campaign, "tool_runtime", None)
        native_workflow = getattr(runtime, "workflow", None)
        if native_workflow is not None:
            if native_workflow.executor is not self.executor:
                raise ValueError("Controller Executor differs from the selected native execution owner")
            # Recovery uses the same native ExecutionContext (including its
            # privately supplied transport credentials), never a fresh route
            # guessed from the controller process environment.
            self._recovery_executor = self.executor.bind(native_workflow.execution_context)

    def run(self, campaign):
        self._campaign = campaign
        self._bind_recovery(campaign)
        state = self.state
        with state.owned():
            if state.data["phase"] == "finished":
                if state.data["prepared"] is None:
                    raise RecoveryRequired("Completed Schema root has no retained prepared identity")
                self._check_prepared(campaign)
                return state.data["finalize"]["reason"]
            try:
                if self.sota is not None:
                    # Source/device facts are checked before preparation can
                    # invoke a model. A rejected SOTA never falls back to design.
                    from .extract_sources import snapshot_sources, validate_target_gpu
                    if self.sota.target_hardware != campaign.sandbox_hardware:
                        raise ValueError("SOTA target facts differ from the selected native environment")
                    validate_target_gpu(self.sota.target_gpu, campaign.sandbox_hardware)
                    source = snapshot_sources(self.directory / "sota-input", self.sota.descriptor,
                                              target_gpu=self.sota.target_gpu)
                    source.require_same_gpu()
                    self._sota_snapshot = source.ref()
                # Recovery precedes scope installation and any new side effect.
                state.reconcile(self._recovery_executor, recover_agent=self.recover_agent)
                if state.data.get("retry_after"):
                    self._backoff(max(0, min(60, state.data["retry_after"] - time.time())))
                with ExitStack() as stack:
                    self._scopes(stack, campaign)
                    from aka.workflow.light.orchestration.infrastructure_retry import bounded_infrastructure
                    stack.enter_context(bounded_infrastructure())
                    try:
                        self._prepare(campaign)
                    except BudgetDenied as error:
                        return self._handle_budget(campaign, error)
                    while True:
                        try:
                            if self.stop_event is not None and self.stop_event.is_set():
                                state.diagnose("requested-interruption", "Controller stopped; retained Episode binding is preserved"
                                    if state.data["binding"] is not None else "Controller stopped at a settled boundary")
                                return "interrupted: requested"
                            if state.data["phase"] == "finalize":
                                return self._finish(campaign, state.data["finalize"]["reason"])
                            state.require_new_work()
                            if state.data["phase"] in {"design", "extract", "optimize"}:
                                self._plan(campaign)
                            elif state.data["phase"] == "implement":
                                self._implement(campaign)
                            else:
                                raise RecoveryRequired("Unknown retained controller phase")
                        except BudgetDenied as error:
                            stopped = self._handle_budget(campaign, error)
                            if stopped is not None:
                                return stopped
            except BaseException as error:
                state.diagnose("recovery-required", f"{type(error).__name__}: {error}")
                raise

    def _handle_budget(self, campaign, error):
        self.state.diagnose("budget-denied", str(error))
        if self.state.pending():
            raise RecoveryRequired("Budget exhausted with unfinished native work; reconcile the retained Episode") from error
        if self.state.data["binding"] is not None:
            self._stop_budget_episode(campaign, str(error))
        if str(error).startswith("window budget:"):
            self._close_window(status="bounded", stop_reason=str(error))
            return None
        return self._finish(campaign, str(error))

    def _prepare(self, campaign):
        data = self.state.data
        if data["prepared"] is not None:
            self._check_prepared(campaign)
            self._refresh_preparation_measurements(campaign)
            return
        self.state.action("prepare")
        public, snapshot = self._retry(lambda: campaign.prepare_native())
        text = _public_text(public, operator_files=getattr(getattr(campaign.problem, "agent", None), "files", ()))
        source = campaign.candidate_workspace.read_source(campaign.workspace)
        source_ref = self.state.retain_source(source, owner=campaign.candidate_workspace,
            origin={"role": "prepared-seed", "verified": True, "design": None})
        records = campaign.experiment_records()
        self._require_records_settled(records)
        if self.state.pending():
            raise RecoveryRequired("Preparation left unresolved native calls")
        if self.initial is not None:
            initial = self.initial
            if isinstance(initial, SuppliedStart):
                initial = initial.bind(campaign=campaign, snapshot=snapshot, source=source,
                    directory=self.directory, materials_snapshot=self.planner.materials_snapshot)
            artifact = read_design(initial.design)
            if (not initial.evidence_refs or initial.prepared_id != snapshot.identity
                    or artifact.manifest["prepared_id"] != snapshot.identity
                    or initial.hardware != campaign.sandbox_hardware
                    or initial.source_identity != source_ref["identity"]):
                raise ValueError("Existing SOTA kernel/Schema must match the prepared Task, actual source and target device")
        data["prepared"] = {"identity": snapshot.identity, "directory": str(snapshot.directory),
            "public_task": text, "public_digest": digest(text), "hardware": campaign.sandbox_hardware,
            "native_policy": self._native_policy(campaign), "source_identity": source_ref["identity"]}
        data["champion"] = source_ref
        data["working_draft"] = source_ref
        data["seen_records"] = [_record_key(record) for record in records]
        data["preparation_measurements"] = self._preparation_feedback(campaign, records[-8:],
            prepared_id=snapshot.identity, source_identity=source_ref["identity"])
        if self.initial is not None:
            data["design"], data["design_version"] = initial.design.as_dict(), artifact.manifest["version"]
            hypothesis = "initial:" + artifact.manifest["closure_digest"]
            data["hypothesis"] = hypothesis
            data["hypotheses"][hypothesis] = self._new_hypothesis(source_ref)
            data["phase"] = "optimize"  # No extract and no spurious initial D window.
        else:
            data["phase"] = "extract" if self.sota is not None else "design"
        data["action"] = None
        self.state.save()

    @staticmethod
    def _preparation_feedback(campaign, records, *, prepared_id, source_identity):
        summaries = []
        for record in records:
            measurement = _measurement_feedback(record)
            identities = sorted({campaign.candidate_workspace.source_identity(campaign.experiment_source(artifact))
                for artifact in record.artifacts if artifact.role == "candidate-source"})
            measurement.update(candidate_source_identities=identities, prepared_id=prepared_id,
                prepared_source_match=identities == [source_identity])
            summaries.append(measurement)
        return summaries

    def _refresh_preparation_measurements(self, campaign):
        """Reproject only retained preparation owners; never dispatch or alter usage.

        Old roots may lack the step summaries. Their original seed is retained
        by source role, independently of the current champion or later trials.
        A frozen Planner request remains unchanged until its operation settles.
        """
        data = self.state.data
        previous = data.get("preparation_measurements", [])
        if not previous:
            return
        prepared = data["prepared"]
        source_identity = prepared.get("source_identity")
        if source_identity is None:
            refs = [data["champion"], data["working_draft"],
                *(hypothesis["seed"] for hypothesis in data["hypotheses"].values())]
            seeds = {ref["identity"]: ref for ref in refs if ref is not None
                and ref["origin"].get("role") == "prepared-seed"}
            if len(seeds) != 1:
                raise RecoveryRequired("Original preparation source identity is unavailable")
            seed, = seeds.values()
            self.state.read_source(seed, owner=campaign.candidate_workspace)
            source_identity = seed["identity"]
        selected = self._preparation_records(campaign)
        if self._preparation_dispatch_gaps(selected):
            raise RecoveryRequired("Original preparation dispatched without admission; explicit historical usage repair is required")
        summaries = self._preparation_feedback(campaign, selected,
            prepared_id=prepared["identity"], source_identity=source_identity)
        if summaries != previous or "source_identity" not in prepared:
            prepared["source_identity"] = source_identity
            data["preparation_measurements"] = summaries
            self.state.save()

    def _preparation_records(self, campaign):
        if str(campaign.workspace.absolute()) != self.state.config["workspace"]:
            raise RecoveryRequired("Preparation recovery belongs to another native workspace")
        records = {_record_key(record): record for record in campaign.experiment_records()}
        try:
            selected = [records[digest(row["experiment"])]
                for row in self.state.data.get("preparation_measurements", [])]
        except KeyError as error:
            raise RecoveryRequired("An original preparation Experiment record disappeared") from error
        self._require_records_settled(selected)
        return selected

    def _preparation_dispatch_gaps(self, records):
        """Match original dispatch receipts, refusing to infer missing launches."""
        data = self.state.data
        corrections = data.get("preparation_dispatch_corrections", {})
        gaps = {}
        for record in records:
            for attempt in record.attempts:
                result = attempt.result
                steps = [step for step in record.steps if step.task_id == attempt.task_id]
                if len(steps) != 1 or record.ref.run_id != str(Path(self.state.config["workspace"]).resolve()):
                    raise RecoveryRequired("Original preparation dispatch belongs to another native owner")
                step, = steps
                owner = {"workspace": record.request["workspace"], "operation": step.operation,
                    "hardware": step.hardware, "timeout": step.timeout, "command_digest": digest(list(step.argv))}
                if attempt.disposition == "not_submitted":
                    if (result is None or result.facts.disposition != "not_submitted"
                            or result.facts.local_stopped is not True or result.job_references
                            or result.exit_code is not None or result.file_receipts):
                        raise RecoveryRequired("Original preparation has no settled pre-submission evidence")
                    owners = []
                    for permit, call in data["calls"].items():
                        if (call["kind"] != "execution" or call["owner"] != owner
                                or call["state"] != "settled" or call.get("disposition") != "not_submitted"
                                or call.get("transport_status") != result.transport_status or call["job_references"]):
                            continue
                        # Match the original observation, including ordinary Eval
                        # with replay disabled. Older live Eval roots retained an
                        # equivalent native-attempt/submission association.
                        live = call.get("live_eval", {})
                        if len(result.attempts) == 1:
                            physical, = result.attempts
                            observed = (call.get("execution_attempts") == [execution_attempt_identity(physical)]
                                and call.get("recorded_eval") == {"attempt_id": attempt.attempt_id,
                                    "idempotency_key": physical.idempotency_key})
                            linked = ("execution_attempts" not in call and "recorded_eval" not in call
                                and live.get("attempt_id") == attempt.attempt_id
                                and call.get("original_permit") == permit
                                and physical.resource is not None and physical.resource.backend == "gateway"
                                and physical.idempotency_key and physical.request_identity
                                and live.get("idempotency_key") == physical.idempotency_key
                                and live.get("request_identity") == physical.request_identity)
                            if ((observed or linked) and physical.attempt_id and physical.disposition == "not_submitted"
                                    and physical.operation == step.operation and physical.exit_code is None
                                    and physical.resource is not None and physical.resource.identifier is None):
                                owners.append(permit)
                        elif not result.attempts:
                            # Local failures without physical receipts still need
                            # the existing explicit original-owner settlement.
                            proof = data.get("unsubmitted_execution_settlements", {}).get(permit, {}).get("evidence")
                            receipts = [item for item in record.artifacts if item.role == "unsubmitted-owner-receipt"]
                            if (len(receipts) == 1 and proof == {"experiment": _record_ref(record),
                                    "attempt_id": attempt.attempt_id, "receipt_sha256": receipts[0].sha256}
                                    and "original-owner-confirmed-not-submitted" in result.facts.basis
                                    and any(item.code == "original-owner-unsubmitted"
                                        and item.details.get("receipt_sha256") == receipts[0].sha256
                                        for item in result.diagnostics)):
                                owners.append(permit)
                    if len(owners) != 1:
                        raise RecoveryRequired("Original preparation rejection has no unique settled admission")
                    continue  # The original admission already consumed its dispatch.
                if (attempt.disposition != "terminal" or result is None
                        or result.facts.disposition != "terminal" or not result.facts.local_stopped
                        or not result.attempts or len(result.job_references) != 1):
                    raise RecoveryRequired("Original preparation dispatch count is unknown; inspect its original owner")
                reference, = result.job_references
                if reference.operation != step.operation:
                    raise RecoveryRequired("Original preparation dispatch identity is unknown; inspect its original owner")
                jobs = [asdict(reference)]
                owners = [(permit, call) for permit, call in data["calls"].items() if call["kind"] == "execution"
                    and jobs == call["job_references"]]
                if owners:
                    if (any(call["owner"] != owner or call["state"] != "settled"
                            or call.get("disposition") != "terminal" for _, call in owners)
                            or not self._preparation_admission_matches(attempt, reference, owners)
                            or any(row["evidence"]["job_reference"] in jobs for row in corrections.values())):
                        raise RecoveryRequired("Original preparation dispatch has ambiguous admission ownership")
                    continue
                # Historical corrections still require exactly one evidenced
                # dispatch. Query receipts never establish missing consumption.
                if len(result.attempts) != 1:
                    raise RecoveryRequired("Original preparation dispatch count is unknown; inspect its original owner")
                physical, = result.attempts
                if (physical.disposition != "terminal" or not physical.attempt_id
                        or physical.resource is None or physical.resource.backend != reference.backend
                        or physical.resource.identifier != reference.identifier or physical.operation != reference.operation):
                    raise RecoveryRequired("Original preparation dispatch identity is unknown; inspect its original owner")
                job = asdict(reference)
                key = digest({"job": job, "physical_attempt": physical.attempt_id})
                proof = {"experiment": _record_ref(record), "attempt_id": attempt.attempt_id,
                    "task_id": attempt.task_id, "physical_attempt_id": physical.attempt_id, "job_reference": job}
                if key in corrections:
                    if corrections[key]["evidence"] != proof:
                        raise RecoveryRequired("Historical preparation dispatch evidence changed")
                else:
                    if key in gaps and gaps[key] != proof:
                        raise RecoveryRequired("Original preparation dispatch has ambiguous Experiment ownership")
                    gaps[key] = proof
        return gaps

    @staticmethod
    def _preparation_admission_matches(attempt, reference, owners):
        """Check native same-Job recovery, without rewriting earlier unknowns."""
        result = attempt.result
        dispatches, queries = [], []
        for physical in result.attempts:
            if (physical.resource is None or physical.resource.backend != reference.backend
                    or physical.resource.identifier != reference.identifier
                    or physical.disposition not in {"terminal", "unknown"}
                    or physical.idempotency_key != reference.idempotency_key
                    or (physical.request_identity or "") != reference.request_identity):
                return False
            if physical.operation == reference.operation and physical.attempt_id and not queries:
                dispatches.append(physical)
            elif physical.operation == "query" and physical.attempt_id is None:
                queries.append(physical)
            else:
                return False
        if not dispatches or result.attempts[-1].disposition != "terminal":
            return False
        if queries:
            # This is the native _resume_results contract: original known Job,
            # original dispatch history, then a confirming terminal GET.
            if "confirmed-job-query" not in result.facts.basis:
                return False
        elif any(item.disposition != "terminal" for item in dispatches):
            return False
        if len(dispatches) == len(owners) == 1:
            call = owners[0][1]
            observed = call.get("execution_attempts")
            recorded = call.get("recorded_eval")
            if recorded is not None and observed is None:
                return False
            return ((observed is None or observed == [execution_attempt_identity(dispatches[0])])
                and (recorded is None or recorded == {"attempt_id": attempt.attempt_id,
                    "idempotency_key": dispatches[0].idempotency_key}
                    # Historical launchers can lack physical key receipts.
                    or not dispatches[0].idempotency_key and recorded.get("attempt_id") == attempt.attempt_id))
        # Multiple paid submissions are legal only within the original live
        # Eval's exact archived association. A shared JobRef alone is insufficient.
        physical_ids = [item.attempt_id for item in dispatches]
        originals = {call.get("original_permit") for _, call in owners}
        if len(originals) != 1:
            return False
        first_permit, = originals
        first = dict(owners).get(first_permit)
        if first is None:
            return False
        live = first.get("live_eval", {})
        if (len(dispatches) != len(owners) or len(set(physical_ids)) != len(physical_ids)
                or reference.backend != "gateway" or not reference.endpoint
                or not reference.idempotency_key or not reference.request_identity
                or live.get("attempt_id") != attempt.attempt_id
                or live.get("idempotency_key") != reference.idempotency_key
                or live.get("request_identity") != reference.request_identity
                or live.get("route_identity") != digest([reference.endpoint, "/v1/jobs/eval", "", ""])
                or not live.get("namespace_identity") or reference.profile):
            return False
        association = {**live, "job_reference": asdict(reference),
            "physical_attempts": physical_ids, "terminal": not bool(queries)}
        observed_prefixes = set()
        for permit, call in owners:
            if (call.get("live_eval") != live or call.get("live_eval_association") != association
                    or call.get("original_permit") != first_permit or call["action"] != first["action"]):
                return False
            if "execution_attempts" in call or "recorded_eval" in call:
                observed = call.get("execution_attempts", [])
                if (call.get("recorded_eval") != {"attempt_id": attempt.attempt_id,
                        "idempotency_key": reference.idempotency_key}
                        or not 0 < len(observed) <= len(dispatches)
                        or permit == first_permit and len(observed) != 1
                        or [item["attempt_id"] for item in observed] != physical_ids[:len(observed)]):
                    return False
                observed_prefixes.add(len(observed))
                for item, physical in zip(observed, dispatches):
                    expected = execution_attempt_identity(physical)
                    if item != expected and item != {**expected, "disposition": "unknown",
                            "resource": {"backend": reference.backend, "identifier": None}}:
                        return False
        if observed_prefixes and observed_prefixes != set(range(1, len(dispatches) + 1)):
            return False
        return all(any(item.category == "recovery" and item.code == "live-eval-association"
            and item.details.get("receipt") == json.dumps(association, sort_keys=True, separators=(",", ":"))
            for item in physical.diagnostics) for physical in dispatches)

    def _repair_preparation_dispatches(self, campaign, reason):
        self._check_prepared(campaign)
        gaps = self._preparation_dispatch_gaps(self._preparation_records(campaign))
        data = self.state.data
        for key, proof in gaps.items():
            data.setdefault("preparation_dispatch_corrections", {})[key] = {
                "kind": "historical-unadmitted", "evidence": proof, "reason": reason,
                "cause": "legacy-scope-bound-before-installation",
                "previous_dispatches": data["counts"]["dispatches"], "consumed_dispatches": 1}
            # This records an evidenced historical consumption, not a fabricated
            # admission. It can exceed a finite limit; future admission must stop.
            data["counts"]["dispatches"] += 1
        if gaps:
            self.state.save()

    def _check_prepared(self, campaign):
        prepared = self.state.data["prepared"]
        snapshot = campaign.problem_provider.snapshot(campaign.problem)
        if snapshot.identity != prepared["identity"]:
            raise RecoveryRequired("Prepared Task identity changed during recovery")
        if prepared["native_policy"] != self._native_policy(campaign):
            raise RecoveryRequired("Native target, evaluation or Episode policy changed during recovery")

    @staticmethod
    def _new_hypothesis(seed):
        return {"seed": seed, "stagnant": 0, "milestones": [], "seen_sources": [], "best_legal": None}

    @staticmethod
    def _native_policy(campaign):
        names = ("framework", "platform", "sandbox_hardware", "sandbox_timeout", "verify_repeats",
                 "verify_run_timeout", "min_improvement_pct", "optimization_mode", "max_iters",
                 "token_budget", "handoff_resumes", "framework_baseline", "convert_after")
        return {**{name: getattr(campaign, name, None) for name in names},
            "episode_timeout_seconds": campaign.module_options.episode_timeout_seconds}

    def _retry(self, operation):
        from aka.workflow.light.orchestration.infrastructure_retry import InfrastructureUnavailable
        failures = self.state.data.get("infrastructure_failures", 0)
        if self.state.data.get("infrastructure_exhausted"):
            raise RecoveryRequired("Infrastructure retries exhausted; record a repair before resuming")
        while True:
            try:
                result = operation()
                self.state.data["infrastructure_failures"] = 0
                self.state.save()
                return result
            except InfrastructureUnavailable as error:
                failures += 1
                self.state.data["infrastructure_failures"] = failures
                exhausted = failures > len(self.limits.retry_delays)
                self.state.data["infrastructure_exhausted"] = exhausted
                self.state.diagnose("temporary-infrastructure", str(error))
                self.state.reconcile(self._recovery_executor, recover_agent=self.recover_agent)
                self._capture_unresolved(self._campaign)
                if exhausted:
                    raise RecoveryRequired("Infrastructure retry allowance exhausted; original action retained") from error
                self._backoff(self.limits.retry_delays[failures - 1])

    def _backoff(self, delay):
        self.state.data["retry_after"] = time.time() + delay
        self.state.save()
        if delay:
            self.sleep(delay)
        self.state.data.pop("retry_after", None)
        self.state.save()

    def _restore_context(self, frozen, *, operation_id=None):
        raw = frozen.get("execution_context")
        if raw is None:
            return None  # Historical operations never acquire new inputs.
        expected = frozen["execution_context_sha256"]
        if operation_id is not None:
            intent = self.directory / "operations" / operation_id / "intent.json"
            if intent.exists():
                original = json.loads(read_regular_bytes(intent, limit=2 * 1024 * 1024))
                if original.get("inputs", {}).get("task/execution.json") != expected:
                    raise RecoveryRequired("Frozen execution context differs from the original Planner intent")
        context = restore_execution_context(raw.encode(), expected_sha256=expected)
        prepared = self.state.data["prepared"]
        context.validate_request(prepared["identity"], prepared["public_task"], prepared["hardware"])
        if context.as_dict()["binding"]["native_policy_sha256"] != digest(prepared["native_policy"]):
            raise RecoveryRequired("Frozen execution context belongs to another native policy")
        return context

    def _new_context(self, campaign, operation_id):
        from ._native import execution_context
        from aka.contracts.experiment import ExperimentRef
        data = self.state.data
        native_workflow = getattr(getattr(campaign, "tool_runtime", None), "workflow", None)
        retained_job = getattr(self, "retained_job_evidence", None)
        task_review = getattr(self, "task_constraint_review", None)
        if native_workflow is None and retained_job is None and task_review is None:
            return None  # Legacy custom embeddings retain their original API.
        self._check_prepared(campaign)
        retained = data.get("execution_context")
        if retained is not None:
            context = self._restore_context(retained)
            document = context.as_dict()
            same_evidence = retained_job is None or all(document[name]["evidence_sha256"] == retained_job.raw_sha256
                for name in ("architecture", "toolchain"))
            same_review = task_review is None or task_review.inspect(data["prepared"]["public_task"]) == document["task_constraints"]
            same_review = same_review and (task_review is None or task_review.bind_device_label(context).sha256 == context.sha256)
            if same_evidence and same_review:
                return context
        return execution_context(campaign=campaign, prepared=data["prepared"], executor=self._recovery_executor,
            directory=self.directory / "execution-inputs" / operation_id,
            task_review=task_review, retained_job=retained_job,
            preparation_records=tuple(ExperimentRef(**row["experiment"])
                for row in data.get("preparation_measurements", [])))

    @staticmethod
    def _freeze_context(context):
        if context is None:
            return {}
        return {"execution_context": context.as_bytes().decode(), "execution_context_sha256": context.sha256}

    def _plan(self, campaign):
        from .planner import PlannerRequest
        data = self.state.data
        self._require_records_settled(campaign.experiment_records())
        if data.get("planner_failures", 0) >= self.limits.planner_attempts:
            raise RecoveryRequired("Planner repair/replan attempts exhausted; repair the retained action before resuming")
        # Do not spend E when an explicitly finite root cannot fund any D.
        self.state.require_allowance("windows")
        self.state.require_allowance("episodes")
        # A deterministic lower bound, not a promise that arbitrary repairs
        # finish: writer, one Coder and one qualification; report and ABBA.
        if data["action"] is None:
            self.state.require_allowance("agent_calls", 3)
            self.state.require_allowance("dispatches", 2)
        hypothesis = data["hypotheses"].get(data["hypothesis"], {})
        operation = data["phase"]
        operation_id = self.state.action(operation)
        frozen = data.get("planner_request")
        if frozen is None or frozen["operation_id"] != operation_id:
            frozen = {"operation_id": operation_id, "version": data["design_version"] + 1,
                "current_design": data["design"], "feedback": [{"text": build_feedback(self.state,
                    qualification_feedback=self._qualification_feedback(campaign))}] if data["design"] else [],
                "require_new": needs_new_hypothesis(hypothesis),
                "sources": {role: data[role] for role in ("champion", "tested_candidate", "working_draft")
                            if data[role] is not None},
                # Freeze the role, reusing the original source refs/bytes. Its
                # absence distinguishes old operations with champion semantics.
                "replacement_seed_role": "working_draft" if data["working_draft"] is not None else "champion"}
            if operation != "extract":
                context = self._new_context(campaign, operation_id)
                frozen.update(self._freeze_context(context))
                if frozen["current_design"] is not None and self.task_constraint_review is not None:
                    try:
                        correction = authorize_contract_correction(context, DesignRef.from_dict(frozen["current_design"]))
                    except ValueError:
                        # Unknown review or inapplicable evidence gives no
                        # special authority; ordinary strict equality remains.
                        correction = None
                    if correction is not None:
                        frozen["contract_correction_sha256"] = correction.sha256
            data["planner_request"] = deepcopy(frozen)
            self.state.save()  # Before role intent/admission, including crash recovery.
        context = self._restore_context(frozen, operation_id=operation_id)
        correction = None
        if frozen.get("contract_correction_sha256"):
            correction = authorize_contract_correction(context, DesignRef.from_dict(frozen["current_design"]))
            if correction.sha256 != frozen["contract_correction_sha256"]:
                raise RecoveryRequired("Frozen Contract correction changed during replay")
        source_files = {}
        for role, ref in frozen["sources"].items():
            source = self.state.read_source(ref, owner=campaign.candidate_workspace)
            source_files.update({f"sources/{role}/{name}": raw for name, raw in source.files.items() if raw is not None})
        if "replacement_seed_role" in frozen:
            role = frozen["replacement_seed_role"]
            if role not in ("working_draft", "champion") or role not in frozen["sources"]:
                raise RecoveryRequired("Frozen replacement seed role has no retained source")
            replacement_seed = frozen["sources"][role]
        else:
            replacement_seed = data["champion"]  # Replay the original legacy rule.
        request = PlannerRequest(prepared_id=data["prepared"]["identity"], operation_id=operation_id,
            version=frozen["version"], directory=self.directory / "operations" / operation_id,
            public_task=data["prepared"]["public_task"], target_hardware=data["prepared"]["hardware"],
            current_design=DesignRef.from_dict(frozen["current_design"]) if frozen["current_design"] else None,
            feedback=tuple(frozen["feedback"]), require_new=frozen["require_new"], source_files=source_files,
            execution_context=context, contract_correction=correction)
        if operation == "extract":
            if self.sota is None:
                raise RecoveryRequired("retained extract phase has no explicit SOTA source")
            from .extract_stages import ExtractRequest
            extraction = ExtractRequest(request.prepared_id, request.operation_id, request.version,
                request.directory, request.public_task, request.target_hardware,
                self.sota.descriptor, self.sota.target_gpu, self._sota_snapshot)
            result = self._retry(lambda: extract(extraction, planner=self.planner))
        else:
            result = self._retry(lambda: getattr(self.planner, operation)(request))
        self.state.require_new_work()
        if result.operation_id != operation_id:
            raise RecoveryRequired("Planner result belongs to another operation")
        error = None
        fact_correction = False
        implementation_context = context
        if result.status == "replace":
            artifact = read_design(result.design_ref)
            if (artifact.manifest["prepared_id"] != request.prepared_id
                    or artifact.manifest["operation_id"] != operation_id
                    or artifact.manifest["version"] != request.version
                    or artifact.manifest["materials_snapshot"] != snapshot_ref(self.planner.materials_snapshot)
                    or artifact.manifest["parent"] != (request.current_design.as_dict() if request.current_design else None)):
                raise RecoveryRequired("Planner design publication binding differs from its request")
            previous = read_design(request.current_design) if request.current_design else None
            if result.contract_correction is not None:
                from .planner import _execution_view
                validate_contract_correction(correction, context, request.current_design)
                check_contract_change(previous, artifact.checked, context=context, proof=correction)
                metadata = result.contract_correction
                fact_correction = _execution_view(previous.checked) == _execution_view(artifact.checked)
                if (metadata != {"kind": "fact_correction" if fact_correction else "execution_rewrite",
                        "proof_sha256": correction.sha256, "execution_context_sha256": context.sha256,
                        "parent_identity": request.current_design.identity}
                        or fact_correction and result.changed_execution_decisions):
                    raise RecoveryRequired("Planner correction metadata differs from the verified publication")
            same = previous is not None and previous.manifest["closure_digest"] == artifact.manifest["closure_digest"]
            new_hypothesis = bool(result.hypothesis_key and result.changed_execution_decisions and result.evidence
                                  and result.hypothesis_key != data["hypothesis"] and not same)
            if same:
                error = "Identical design bytes are keep; replacement cannot renew a hypothesis"
            elif not fact_correction and needs_new_hypothesis(data["hypotheses"].get(result.hypothesis_key, {})):
                error = "Revisiting an exhausted hypothesis retains its earlier history"
            elif not fact_correction and request.require_new and not new_hypothesis:
                error = "Current hypothesis is exhausted; provide a new execution decision with evidence"
            else:
                data["design"], data["design_version"] = result.design_ref.as_dict(), request.version
                if not fact_correction and (new_hypothesis or data["hypothesis"] is None):
                    key = result.hypothesis_key or "design:" + artifact.manifest["closure_digest"]
                    data["hypothesis"] = key
                    data["hypotheses"].setdefault(key, self._new_hypothesis(replacement_seed))
                # Continue the frozen branch source across design changes.
                # A seed transfers no qualification, measurement or promotion.
                if not fact_correction:
                    data["working_draft"] = replacement_seed
        elif result.status == "keep":
            if (request.require_new or data["working_draft"] is None or not result.next_actions
                    or data["design"] is None):
                error = "Keep requires an unexhausted hypothesis, real draft and a testable next action"
            elif hypothesis.get("uncertain_windows", 0) and digest(list(result.next_actions)) in hypothesis.get("tried_actions", []):
                error = "Unknown performance requires an untried evidence-gathering action or a new hypothesis"
            else:
                try:
                    retained = read_design(request.current_design)
                    published_context = publication_execution_context(retained)
                    if published_context is not None:
                        if context is None:
                            raise ValueError("keep cannot discard published context")
                        old, new = published_context.as_dict(), context.as_dict()
                        if old["binding"] != new["binding"] or any(
                                new[name]["source"] != "unknown" and new[name]["value"] != old[name]["value"]
                                for name in ("architecture", "toolchain", "selected_framework")):
                            raise ValueError("new planning facts contradict the published design")
                        # A new evidence digest alone is not a new design. E's
                        # input remains frozen on its own operation; D/reviewer
                        # retain the original publication's exact context.
                        implementation_context = published_context
                    else:
                        check_context_hardware(retained.checked, context, parent=retained)
                    validate_design_execution_context(retained, implementation_context)
                except ValueError:
                    error = "Conflicting execution facts require a corrected publication before implementation"
        elif result.status == "stop":
            # Planner may close its own hypothesis, never the user's continuous
            # task. A no-design response enters bounded replanning/diagnosis.
            error = "Planner ended its hypothesis; a new evidence-backed design is required"
        else:
            error = "Planner did not publish a valid decision: " + json.dumps(result.diagnostics, ensure_ascii=False)
        if error is not None:
            failures = data.get("planner_failures", 0) + 1
            data["planner_failures"] = failures
            if failures < self.limits.planner_attempts:
                data["action"] = None
            self.state.diagnose("planner-replan", error)
            if failures >= self.limits.planner_attempts:
                raise RecoveryRequired("Planner repair/replan attempts exhausted; inspect retained operation drafts")
            self._backoff(self.limits.retry_delays[min(failures - 1, len(self.limits.retry_delays) - 1)]
                          if self.limits.retry_delays else 0)
            return
        data["planner_failures"] = 0
        if implementation_context is not None:
            data["execution_context"] = self._freeze_context(implementation_context)
        data["next_actions"] = list(result.next_actions)
        if data["hypothesis"] is not None and not fact_correction:
            actions = data["hypotheses"][data["hypothesis"]].setdefault("tried_actions", [])
            key = digest(data["next_actions"])
            if key not in actions:
                actions.append(key)
        data["phase"], data["action"] = "implement", None
        self.state.save()

    def _open_window(self):
        data = self.state.data
        if data["window"] is None:
            self.state.require_allowance("windows")
            data["counts"]["windows"] += 1
            data["window"] = {"id": data["counts"]["windows"], "design": data["design"],
                "hypothesis": data["hypothesis"], "episodes": [], "progress": [],
                "start_counts": dict(data["counts"])}
            self.state.save()
        return data["window"]

    def _project_qualification(self, binding, receipt, sealed, source):
        if receipt is None:
            return unavailable("qualification-receipt-missing", episode=binding["episode"])
        if not isinstance(receipt, dict):
            return unavailable("qualification-unknown", episode=binding["episode"])
        if sealed is None or source is None:
            return unavailable("qualification-sealed-source-unavailable", episode=binding["episode"])
        try:
            artifact = read_design(DesignRef.from_dict(binding["design"]))
        except (KeyError, TypeError, ValueError, OSError):
            # A missing public reading view cannot turn a native settlement
            # into another review or block retention of its original verdict.
            return unavailable("qualification-design-unavailable", episode=binding["episode"])
        if artifact.manifest["prepared_id"] != binding["prepared_id"]:
            return unavailable("qualification-binding-mismatch", episode=binding["episode"])
        obligations = {"root.CONTRACT.Algorithm_requirement", "root.CONTRACT.Hardware_specification", "dispatch"}
        obligations.update(name + ".DESIGN." + region for name in artifact.checked.resolved.raw_task
            for region in ("Math_reshape", "Work_schedule", "Tile_pipeline"))
        return project(receipt, binding=binding, prepared=self.state.data["prepared"],
            sealed_identity=sealed["identity"], source_files=source.files,
            design_files=artifact.checked.files, obligations=obligations)

    def _qualification_feedback(self, campaign, *, seed_identity=None, context_binding=None):
        """New request view only: legacy rows and native receipts stay untouched."""
        from aka.workflow.light.long_horizon.store import CampaignStore
        data = self.state.data
        expected = data.get("execution_context", {}) if context_binding is None else context_binding
        fallback = unavailable("qualification-no-matching-review" if data["episodes"]
            else "qualification-receipt-missing")
        native = None
        for row in reversed(list(data["episodes"].values())[-WINDOW_EPISODES:]):
            binding, sealed = row["binding"], row.get("sealed_candidate")
            if binding["design"]["identity"] != data["design"]["identity"]:
                continue
            if seed_identity is not None and (sealed is None or sealed["identity"] != seed_identity):
                continue
            summary = row.get("qualification_feedback")
            if summary is None:
                # Borrow the existing native archive reader without opening an
                # engine/runtime or exposing its private archive to the Agent.
                if native is None:
                    checkpoint, _ = CampaignStore.checkpoint(campaign.workspace)
                    native = {item["episode"]: item for item in checkpoint.attempts}
                attempt = native.get(row["episode"], {})
                try:
                    source = self.state.read_source(sealed, owner=campaign.candidate_workspace) if sealed else None
                except (KeyError, ValueError, OSError, RecoveryRequired):
                    summary = unavailable("qualification-sealed-source-unavailable", episode=row["episode"])
                else:
                    summary = self._project_qualification(binding, attempt.get("candidate_qualification"), sealed, source)
            if summary.get("status") == "reviewed" and not applies_to(summary,
                    prepared=data["prepared"], design_identity=data["design"]["identity"],
                    source_identity=seed_identity,
                    execution_context_sha256=expected.get("execution_context_sha256")):
                summary = unavailable("qualification-binding-mismatch", episode=row["episode"])
            if seed_identity is not None:
                if summary.get("status") == "reviewed":
                    return summary
                if "original_episode" not in fallback:
                    fallback = summary
                continue
            # E sees the newest concrete failure ahead of later Episodes that
            # never produced an independent review, with its original identity.
            if summary.get("verdict") == "fail":
                return summary
            if "original_episode" not in fallback:
                fallback = summary
        return fallback

    def _implement(self, campaign):
        from .planner import coder_context
        data = self.state.data
        window = self._open_window()
        binding = data["binding"]
        if binding is None:
            self.state.require_allowance("episodes")
            self.state.require_allowance("agent_calls")
            self.state.require_window_allowance("agent_calls")
            self.state.require_window_allowance("dispatches")
            number = data["counts"]["episodes"] + 1
            operation_id = self.state.action("implement")
            execution = (self._restore_context(data["execution_context"]) if data.get("execution_context")
                         else self._new_context(campaign, operation_id))
            seed = data["working_draft"] or data["champion"]
            binding = {"episode": number, "operation_id": operation_id, "window": window["id"],
                "design": data["design"], "prepared_id": data["prepared"]["identity"],
                "seed": seed, "hypothesis": data["hypothesis"], **self._freeze_context(execution)}
            frontier = data["hypotheses"][data["hypothesis"]].get("native_frontier")
            context = coder_context(DesignRef.from_dict(data["design"]), data["prepared"]["identity"], number, seed["identity"],
                next_actions=data.get("next_actions", ()),
                comparison_target="kernel-" + frontier["source"]["identity"] if frontier else None,
                execution_context=execution)
            context += coder_feedback(self._qualification_feedback(campaign,
                seed_identity=seed["identity"], context_binding=binding))
            binding.update(context=context, context_id=digest(context))
            if execution is not None:
                # First adoption by a new D also freezes the context for later
                # operations on a legacy/imported design; its old inputs stay intact.
                data["execution_context"] = self._freeze_context(execution)
            data["binding"] = binding
            data["counts"]["episodes"] = number
            self.state.save()  # Before native admission and seed installation.
        seed = self.state.read_source(binding["seed"], owner=campaign.candidate_workspace)
        execution = self._restore_context(binding)
        def qualify(**native):
            source = campaign.candidate_workspace.read_source(native["worktree"].path,
                revision=native["candidate_commit"])
            identity = campaign.candidate_workspace.source_identity(source)
            if self.candidate_qualification is not None:
                result = self.candidate_qualification(binding=dict(binding), source=source,
                    source_identity=identity, **native)
                if execution is not None and (result.get("binding", {}).get("source_identity") != identity
                        or result.get("binding", {}).get("execution_context_sha256") != execution.sha256):
                    raise RecoveryRequired("Qualification callback does not bind the sealed source and execution context")
                return result
            original = self.state.read_source(data["hypotheses"][binding["hypothesis"]]["seed"],
                owner=campaign.candidate_workspace)
            result = self.planner.qualify(design_ref=DesignRef.from_dict(binding["design"]),
                prepared_id=binding["prepared_id"], operation_id=binding["operation_id"] + ":qualification",
                directory=self.directory / "operations" / (binding["operation_id"] + "-qualification"),
                source=source, source_identity=identity,
                evidence={"episode": binding["episode"], "candidate_commit": native["candidate_commit"],
                    "public_task": data["prepared"]["public_task"],
                    "target_hardware": data["prepared"]["hardware"],
                    "seed_identity": campaign.candidate_workspace.source_identity(original),
                    "seed_files": dict(original.files)},
                **({"execution_context": execution} if execution is not None else {}))
            if result.get("binding", {}).get("source_identity") != identity:
                raise RecoveryRequired("Qualification receipt does not bind the sealed native source")
            if result.get("binding", {}).get("execution_context_sha256") != binding.get("execution_context_sha256"):
                raise RecoveryRequired("Qualification receipt does not bind the frozen execution context")
            return {**result, "source_identity": identity}
        engine = campaign.create_engine(continuous=campaign.max_iters == 0,
            episode_context=binding["context"], episode_seed=seed, candidate_qualification=qualify)
        reason = self._retry(lambda: engine.run(episode=binding["episode"]))
        if reason != "episode-settled":
            # Native finite limits remain effective. A rejected fresh Episode
            # has no side effects and can be detached without a fake settlement.
            from aka.workflow.light.long_horizon.store import CampaignStore
            _, active = CampaignStore.checkpoint(campaign.workspace)
            if active is None:
                data["binding"] = None
                data["action"] = None
                self.state.save()
                raise BudgetDenied(reason)
            raise RecoveryRequired("Native Episode returned before settlement: " + reason)
        facts = campaign.episode_facts(binding["episode"])
        self._settle(campaign, binding, facts)
        if len(self.state.data["window"]["episodes"]) == WINDOW_EPISODES:
            self._close_window()

    @staticmethod
    def _require_records_settled(records):
        for record in records:
            if record.lifecycle != "terminal" or any(attempt.disposition not in {"terminal", "not_submitted"}
                                                       for attempt in record.attempts):
                raise RecoveryRequired("Original Experiment owner is unresolved: " + str(_record_ref(record)))

    def _settle(self, campaign, binding, facts):
        data = deepcopy(self.state.data)
        episode = binding["episode"]
        if str(episode) in data["episodes"]:
            return
        attempt = facts["attempt"]
        if facts["active"] is not None or attempt is None or attempt["episode"] != episode:
            raise RecoveryRequired("Native Episode is not archived and settled")
        self._require_records_settled(facts["records"])
        if self.state.pending():
            raise RecoveryRequired("Episode has unresolved native calls")
        owner = campaign.candidate_workspace
        draft = self.state.retain_source(facts["draft"], owner=owner,
            origin={"episode": episode, "role": "working_draft", "design": binding["design"]["identity"], "verified": False})
        champion = self.state.retain_source(facts["champion"], owner=owner,
            origin={"episode": episode, "role": "champion", "design": binding["design"]["identity"] if attempt["accepted"]
                    else data["champion"]["origin"].get("design"), "verified": True})
        tested = data["tested_candidate"]
        records = []
        measurements = []
        fresh = []
        for record in facts["records"]:
            key = _record_key(record)
            if key in data["seen_records"]:
                continue
            records.append(_record_ref(record))
            measurement = _measurement_feedback(record)
            measurement["candidate_source_identities"] = []
            measurements.append(measurement)
            for artifact in record.artifacts:
                if artifact.role == "candidate-source":
                    source = campaign.experiment_source(artifact)
                    tested = self.state.retain_source(source, owner=owner,
                        origin={"episode": episode, "role": "tested_candidate", "design": binding["design"]["identity"],
                                "experiment": _record_ref(record), "artifact": artifact.as_dict()})
                    measurement["candidate_source_identities"].append(tested["identity"])
                    fresh.append((record, tested))
        # Sealing alone is not measurement evidence. Keep a sealed source as a
        # distinct optional reference without claiming it was the last measured.
        sealed = self.state.retain_source(facts["candidate"], owner=owner,
            origin={"episode": episode, "role": "sealed_candidate", "design": binding["design"]["identity"]}) \
            if facts["candidate"] is not None else None
        qualification = attempt.get("candidate_qualification") or {}
        if qualification.get("qualified") is True and (sealed is None or qualification.get("source_identity") != sealed["identity"]):
            raise RecoveryRequired("Progress qualification differs from the sealed native source")
        progress_assessment = {}
        progress = self._progress(binding, attempt, fresh, hypothesis=data["hypotheses"][binding["hypothesis"]],
            assessment=progress_assessment, sealed_source_identity=sealed["identity"] if sealed is not None else None)
        data["champion"], data["tested_candidate"], data["working_draft"] = champion, tested, draft
        data["episodes"][str(episode)] = {"episode": episode, "binding": binding,
            "status": attempt["status"], "accepted": attempt["accepted"], "violation": attempt.get("violation"),
            "tokens": attempt.get("tokens"), "telemetry": attempt.get("telemetry"),
            "draft": draft, "sealed_candidate": sealed, "tested_candidate": tested,
            "champion": champion, "records": records, "measurements": measurements[-8:], "progress": progress,
            "progress_assessment": progress_assessment,
            "qualification_feedback": self._project_qualification(binding,
                attempt.get("candidate_qualification"), sealed, facts["candidate"])}
        data["seen_records"] = sorted(set(data["seen_records"]) | {_record_key(record) for record in facts["records"]})
        data["window"]["episodes"].append(episode)
        data["window"]["progress"].extend(progress)
        data["counts"]["settled_episodes"] += 1
        # Native receipts survive a lost controller callback and retain their
        # explicit measurement quality. Do not turn absent usage into exact 0.
        data["binding"], data["action"] = None, None
        self.state.data = data
        self._sync_native_usage(campaign)
        self.state.save()

    def _stop_budget_episode(self, campaign, reason):
        binding = self.state.data["binding"]
        self._require_records_settled(campaign.experiment_records())
        seed = self.state.read_source(binding["seed"], owner=campaign.candidate_workspace)
        engine = campaign.create_engine(continuous=campaign.max_iters == 0,
            episode_context=binding["context"], episode_seed=seed,
            candidate_qualification=lambda **kwargs: {"qualified": False, "reason": reason})
        engine.stop_episode(episode=binding["episode"], reason=reason)
        self._settle(campaign, binding, campaign.episode_facts(binding["episode"]))

    def _progress(self, binding, attempt, fresh, *, hypothesis=None, assessment=None, sealed_source_identity=None):
        from aka.evaluation.source_archives import source_artifact
        from .progress import comparison_key, measurement_evidence, positive
        import hashlib
        if hypothesis is None:
            hypothesis = self.state.data["hypotheses"][binding["hypothesis"]]
        assessment = assessment if assessment is not None else {}
        assessment.update(status="no-progress", reasons=[])
        qualified = attempt.get("candidate_qualification")
        if qualified is None:
            # Absence does not establish whether review was attempted or failed.
            assessment["reasons"] = ["qualification-receipt-missing"]
            return []
        if not isinstance(qualified, dict):
            assessment["reasons"] = ["qualification-unknown"]
            return []
        receipt = qualified.get("binding")
        if receipt is not None and (not isinstance(receipt, dict)
                or receipt.get("prepared_id") != binding["prepared_id"]
                or receipt.get("design_identity") != binding["design"]["identity"]
                or receipt.get("source_identity") != qualified.get("source_identity")
                or receipt.get("public_task_sha256") != hashlib.sha256(
                    self.state.data["prepared"]["public_task"].encode()).hexdigest()
                or receipt.get("target_hardware_sha256") != hashlib.sha256(
                    self.state.data["prepared"]["hardware"].encode()).hexdigest()
                or receipt.get("execution_context_sha256") != binding.get("execution_context_sha256")):
            assessment["reasons"] = ["qualification-binding-mismatch"]
            return []
        if qualified.get("qualified") is not True:
            # An internally consistent result may still review an older source.
            # Only the actual native seal can bind a negative verdict here.
            identity = qualified.get("source_identity")
            if sealed_source_identity is None or identity is None:
                reason = "qualification-unknown"
            elif identity != sealed_source_identity:
                reason = "qualification-binding-mismatch"
            else:
                reason = "qualification-failed" if qualified.get("verdict") == "fail" else "qualification-unknown"
            assessment["reasons"] = [reason]
            return []
        if qualified.get("substantive_change") is not True:
            assessment["reasons"] = ["no-substantive-change"]
            return []
        identity = qualified.get("source_identity")
        if identity == hypothesis["seed"]["identity"] or identity in hypothesis["seen_sources"]:
            assessment["reasons"] = ["unchanged-or-replayed-source"]
            return []
        selected = [(record, source) for record, source in fresh if source["identity"] == identity
                    and _record_key(record) not in hypothesis.get("progress_records", [])]
        if not selected:
            assessment["reasons"] = ["no-fresh-qualified-record"]
            return []
        campaign = self._campaign
        evidence = []
        with campaign.experiment_record_reader() as records:
            for record, source in selected:
                value = measurement_evidence(records, record, owner=campaign.candidate_workspace,
                    task=campaign.problem.evaluation, hardware=campaign.sandbox_hardware)
                if value is not None and value["source_identity"] == identity:
                    evidence.append((record, source, value))
        if not evidence:
            assessment["reasons"] = ["no-complete-current-task-evidence"]
            return []
        # Collect all this sealed source's records before consuming it. A normal
        # Eval, correctness-only check and ABBA can coexist in one Episode.
        hypothesis["seen_sources"].append(identity)
        hypothesis.setdefault("progress_records", []).extend(_record_key(record) for record, _, _ in evidence)
        progress = []
        def milestone(name):
            if name not in hypothesis["milestones"]:
                hypothesis["milestones"].append(name)
                progress.append(name)
        milestone("first-compiled")
        milestone("first-correct")
        performance = [(record, source, value) for record, source, value in evidence if value["performance"]]
        if performance:
            milestone("first-qualified-full-measurement")
        threshold = float(getattr(campaign, "min_improvement_pct", 0))
        frontier = hypothesis.get("native_frontier")
        if frontier is None and performance:
            seed = self.state.read_source(hypothesis["seed"], owner=campaign.candidate_workspace)
            artifact, _ = source_artifact(seed)
            frontier = {"source": hypothesis["seed"], "artifact_sha256": hashlib.sha256(artifact.content).hexdigest()}
        decisive = False
        for record, source, value in performance:
            latency = value["latency_us"]
            baseline = value["baseline_latency_us"]
            if attempt.get("accepted") is True and value["comparison"]:
                # The archived native owner independently settled promotion.
                # It may compare to a champion different from this design's
                # exploratory frontier; retain that stronger source evidence.
                decisive = True
                if "native-promotion" not in progress:
                    progress.append("native-promotion")
            if (value["comparison"] and value["baseline_sha256"] == frontier["artifact_sha256"]
                    and positive(baseline)):
                # This record compares against the retained best source itself;
                # no latency or environment identity is borrowed from an old job.
                decisive = True
                milestone("first-native-comparison")
                if latency < baseline * (1 - threshold / 100):
                    if "native-comparison-improved" not in progress:
                        progress.append("native-comparison-improved")
                if latency < baseline:
                    frontier = {"source": source, "artifact_sha256": value["candidate_sha256"],
                                "experiment": _record_ref(record)}
            key = comparison_key(value["measurements"])
            if key is None:
                continue
            milestone("first-comparable-performance")
            bests = hypothesis.setdefault("best_by_comparison", {})
            old = hypothesis.get("best_legal")
            if old is not None:
                bests.setdefault(old["comparison_key"], old)
            best = bests.get(key)
            if best is not None:
                decisive = True
                if latency < best["latency_us"] * (1 - threshold / 100):
                    if "comparable-performance-improved" not in progress:
                        progress.append("comparable-performance-improved")
            if best is None or latency < best["latency_us"]:
                bests[key] = {"latency_us": latency, "comparison_key": key,
                             "experiment": _record_ref(record), "source": source}
                hypothesis["best_legal"] = bests[key]
                if best is not None and frontier["source"]["identity"] == best["source"]["identity"]:
                    frontier = {"source": source, "artifact_sha256": value["candidate_sha256"],
                                "experiment": _record_ref(record)}
        if performance:
            if hypothesis.get("native_frontier") is None and frontier["source"]["identity"] == hypothesis["seed"]["identity"]:
                # The first legal implementation is this design's starting
                # point, even when it is slower than the global champion.
                record, source, value = performance[0]
                frontier = {"source": source, "artifact_sha256": value["candidate_sha256"],
                            "experiment": _record_ref(record)}
            hypothesis["native_frontier"] = frontier
            if "native-promotion" in progress:
                record, source, value = next(row for row in performance if row[2]["comparison"])
                hypothesis["native_frontier"] = {"source": source, "artifact_sha256": value["candidate_sha256"],
                                                 "experiment": _record_ref(record)}
        if progress:
            assessment["status"] = "progress"
        elif performance and not decisive:
            assessment.update(status="unknown", reasons=["cross-record-comparison-unavailable"],
                comparison_target="kernel-" + frontier["source"]["identity"],
                next_evidence="Compare the new source against the retained design best in one native ABBA record, "
                              "obtain owner-observed matching environment evidence, or change the hypothesis.")
        else:
            assessment["reasons"] = ["no-new-milestone-or-best-improvement"]
        return progress

    def _close_window(self, *, status="settled", stop_reason=None):
        data = self.state.data
        window = data["window"]
        hypothesis = data["hypotheses"][window["hypothesis"]]
        unknown = any(data["episodes"][str(episode)].get("progress_assessment", {}).get("status") == "unknown"
                      for episode in window["episodes"])
        if window["progress"]:
            hypothesis["stagnant"] = 0
            hypothesis["uncertain_windows"] = 0
        elif unknown:
            hypothesis["uncertain_windows"] = hypothesis.get("uncertain_windows", 0) + 1
        else:
            hypothesis["stagnant"] += 1
        window["progress_status"] = "progress" if window["progress"] else "unknown" if unknown else "no-progress"
        window["uncertain_windows"] = hypothesis.get("uncertain_windows", 0)
        window["stagnant"] = hypothesis["stagnant"]
        window["status"] = status
        if stop_reason:
            window["stop_reason"] = stop_reason
        data["windows"].append(window)
        data["window"], data["phase"], data["action"] = None, "optimize", None
        self.state.save()

    def _finish(self, campaign, reason):
        data = self.state.data
        if self.state.pending() or data["binding"] is not None:
            raise RecoveryRequired("Cannot finalize an unresolved native owner")
        self._require_records_settled(campaign.experiment_records())
        if data["window"] is not None:
            data["window"].update(status="partial", stop_reason=reason)
        if data["action"] is not None and data["action"]["kind"] != "finalize":
            data["action"] = None
        operation_id = self.state.action("finalize")
        data["phase"] = "finalize"
        previous = data["finalize"]
        if previous is not None and previous["reason"] != reason:
            raise RecoveryRequired("Finalization reason differs from retained intent")
        data["finalize"] = {"operation_id": operation_id, "reason": reason, "done": False}
        self.state.save()
        result = campaign.finish_native(reason, operation_id=operation_id)
        if self.state.pending():
            raise RecoveryRequired("Finalization left unresolved native work")
        data["finalize"]["done"] = True
        data["phase"], data["action"] = "finished", None
        self.state.save()
        return result

    def settle_unsubmitted_execution(self, permit, *, campaign, experiment_ref, attempt_id, reason):
        """Settle one consumed permit from its original owner's explicit receipt."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A concrete original-owner diagnosis is required")
        with self.state.owned():
            if str(campaign.workspace.absolute()) != self.state.config["workspace"]:
                raise RecoveryRequired("Execution recovery belongs to another native workspace")
            records = [item for item in campaign.experiment_records() if item.ref == experiment_ref]
            if len(records) != 1:
                raise RecoveryRequired("Original Experiment is unavailable")
            record, = records
            if (record.ref.run_id != str(campaign.workspace.resolve()) or record.lifecycle != "terminal"
                    or len(record.attempts) != 1 or len(record.steps) != 1):
                raise RecoveryRequired("Original Experiment is not a settled single-attempt owner")
            attempt, = record.attempts
            result = attempt.result
            if (attempt.attempt_id != attempt_id or attempt.disposition != "not_submitted" or result is None
                    or result.facts.disposition != "not_submitted" or result.facts.local_stopped is not True
                    or result.attempts or result.job_references or result.exit_code is not None or result.file_receipts
                    or "original-owner-confirmed-not-submitted" not in result.facts.basis):
                raise RecoveryRequired("Original owner has no explicit pre-submission settlement")
            receipts = [item for item in record.artifacts if item.role == "unsubmitted-owner-receipt"]
            if (len(receipts) != 1 or not any(item.code == "original-owner-unsubmitted"
                    and item.details.get("receipt_sha256") == receipts[0].sha256 for item in result.diagnostics)):
                raise RecoveryRequired("Original owner settlement receipt is missing")
            data = self.state.data
            call = data["calls"].get(permit)
            step, = record.steps
            owner = {"workspace": record.request.get("workspace"), "operation": step.operation,
                "hardware": step.hardware, "timeout": step.timeout, "command_digest": digest(list(step.argv))}
            if (call is None or call["kind"] != "execution" or call["owner"] != owner or call["job_references"]):
                raise RecoveryRequired("Original admission and Experiment identities differ")
            proof = {"experiment": experiment_ref.as_dict(), "attempt_id": attempt_id,
                "receipt_sha256": receipts[0].sha256}
            settlements = data.get("unsubmitted_execution_settlements", {})
            if permit in settlements:
                if settlements[permit]["evidence"] != proof or call["state"] != "settled":
                    raise RecoveryRequired("Original execution settlement changed")
                return
            if call["state"] != "unknown":
                raise RecoveryRequired("Execution permit is not awaiting owner reconciliation")
            settlements[permit] = {"evidence": proof, "reason": reason}
            data["unsubmitted_execution_settlements"] = settlements
            call.update(state="settled", disposition="not_submitted", transport_status=result.transport_status)
            self.state.save()

    def resume_after_repair(self, reason, *, preparation_campaign=None):
        """Record explicit repair without refunding usage or inventing admission.

        Supplying the original preparation Campaign explicitly requests a
        historical consumption correction from its retained dispatch receipts.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A concrete repair diagnosis is required")
        if preparation_campaign is not None:
            self._bind_recovery(preparation_campaign)
        with self.state.owned():
            if preparation_campaign is not None:
                self._check_prepared(preparation_campaign)
                if str(preparation_campaign.workspace.absolute()) != self.state.config["workspace"]:
                    raise RecoveryRequired("Preparation recovery belongs to another native workspace")
            self.state.reconcile(self._recovery_executor, recover_agent=self.recover_agent)
            data = self.state.data
            if data["phase"] == "finished":
                raise ValueError("A completed finite root cannot be renewed")
            if preparation_campaign is not None:
                self._repair_preparation_dispatches(preparation_campaign, reason)
            data["infrastructure_failures"] = 0
            data["infrastructure_exhausted"] = False
            data["planner_failures"] = 0
            if data["phase"] in {"design", "extract", "optimize"}:
                data["action"] = None
            self.state.diagnose("supervisor-repair", reason)


def implement(campaign, *, episode, design_context, seed, candidate_qualification):
    """Single-Episode capability; the persistent driver alone owns D/E loops."""
    return campaign.create_engine(continuous=campaign.max_iters == 0, episode_context=design_context,
        episode_seed=seed, candidate_qualification=candidate_qualification).run(episode=episode)
