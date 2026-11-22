"""One persistent Schema controller around the original Light Episode engine."""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Mapping

from aka.core.active_time import ActiveBudget, budget_scope
from aka.core.filesystem import read_regular_bytes
from .materials import DesignRef, read_design, snapshot_ref
from .run_state import BudgetDenied, RecoveryRequired, RunLimits, RunState, digest


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


def build_feedback(state):
    """Bounded public prose/identities, never raw private evaluator inputs."""
    data = state.data
    hypothesis = data["hypotheses"].get(data["hypothesis"], {})
    episodes = list(data["episodes"].values())[-WINDOW_EPISODES:]
    value = {"design": data["design"], "prepared_id": data["prepared"]["identity"],
        "champion": data["champion"], "tested_candidate": data["tested_candidate"],
        "working_draft": data["working_draft"], "stagnant_windows": hypothesis.get("stagnant", 0),
        "milestones": hypothesis.get("milestones", []), "best_legal_measurement": hypothesis.get("best_legal"),
        "preparation_measurements": data.get("preparation_measurements", []),
        "episodes": [{key: row.get(key) for key in ("episode", "status", "accepted", "violation", "progress", "records", "measurements")}
                     for row in episodes],
        "require_new": hypothesis.get("stagnant", 0) >= 2,
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
                 candidate_qualification=None, sleep=time.sleep, sota: SotaStart | None = None):
        if initial is not None and sota is not None:
            raise ValueError("supplied kernel/Schema and source-only extraction are distinct, mutually exclusive C inputs")
        if initial is not None and not isinstance(initial, (ExistingStart, SuppliedStart)) or sota is not None and not isinstance(sota, SotaStart):
            raise TypeError("C input must be an explicit ExistingStart, SuppliedStart or SotaStart")
        self.directory = Path(directory).absolute()
        self.planner, self.executor, self.initial = planner, executor, initial
        self.sota = sota
        self.stop_event, self.recover_agent, self.sleep = stop_event, recover_agent, sleep
        self.candidate_qualification = candidate_qualification
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
                jobs = [asdict(reference) for reference in result.job_references] if result is not None else []
                owners = [call for call in data["calls"].values() if call["kind"] == "execution"
                    and jobs and jobs == call["job_references"]]
                if owners:
                    if (len(owners) != 1 or any(row["evidence"]["job_reference"] in jobs
                                               for row in corrections.values())):
                        raise RecoveryRequired("Original preparation dispatch has ambiguous admission ownership")
                    continue
                if (attempt.disposition != "terminal" or result is None
                        or result.facts.disposition != "terminal" or not result.facts.local_stopped
                        or len(result.attempts) != 1 or len(result.job_references) != 1):
                    raise RecoveryRequired("Original preparation dispatch count is unknown; inspect its original owner")
                physical, = result.attempts
                reference, = result.job_references
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
                "current_design": data["design"], "feedback": [{"text": build_feedback(self.state)}] if data["design"] else [],
                "require_new": hypothesis.get("stagnant", 0) >= 2,
                "sources": {role: data[role] for role in ("champion", "tested_candidate", "working_draft")
                            if data[role] is not None}}
            data["planner_request"] = deepcopy(frozen)
            self.state.save()  # Before role intent/admission, including crash recovery.
        source_files = {}
        for role, ref in frozen["sources"].items():
            source = self.state.read_source(ref, owner=campaign.candidate_workspace)
            source_files.update({f"sources/{role}/{name}": raw for name, raw in source.files.items() if raw is not None})
        request = PlannerRequest(prepared_id=data["prepared"]["identity"], operation_id=operation_id,
            version=frozen["version"], directory=self.directory / "operations" / operation_id,
            public_task=data["prepared"]["public_task"], target_hardware=data["prepared"]["hardware"],
            current_design=DesignRef.from_dict(frozen["current_design"]) if frozen["current_design"] else None,
            feedback=tuple(frozen["feedback"]), require_new=frozen["require_new"], source_files=source_files)
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
        if result.status == "replace":
            artifact = read_design(result.design_ref)
            if (artifact.manifest["prepared_id"] != request.prepared_id
                    or artifact.manifest["operation_id"] != operation_id
                    or artifact.manifest["version"] != request.version
                    or artifact.manifest["materials_snapshot"] != snapshot_ref(self.planner.materials_snapshot)
                    or artifact.manifest["parent"] != (request.current_design.as_dict() if request.current_design else None)):
                raise RecoveryRequired("Planner design publication binding differs from its request")
            previous = read_design(request.current_design) if request.current_design else None
            same = previous is not None and previous.manifest["closure_digest"] == artifact.manifest["closure_digest"]
            new_hypothesis = bool(result.hypothesis_key and result.changed_execution_decisions and result.evidence
                                  and result.hypothesis_key != data["hypothesis"] and not same)
            if same:
                error = "Identical design bytes are keep; replacement cannot renew a hypothesis"
            elif data["hypotheses"].get(result.hypothesis_key, {}).get("stagnant", 0) >= 2:
                error = "Revisiting an exhausted hypothesis retains its earlier history"
            elif request.require_new and not new_hypothesis:
                error = "Current hypothesis is exhausted; provide a new execution decision with evidence"
            else:
                data["design"], data["design_version"] = result.design_ref.as_dict(), request.version
                if new_hypothesis or data["hypothesis"] is None:
                    key = result.hypothesis_key or "design:" + artifact.manifest["closure_digest"]
                    data["hypothesis"] = key
                    data["hypotheses"].setdefault(key, self._new_hypothesis(data["champion"]))
                # Every new graph begins at the real champion. Attribution of
                # this inherited source is retained until actual new evidence.
                data["working_draft"] = data["champion"]
        elif result.status == "keep":
            if (request.require_new or data["working_draft"] is None or not result.next_actions
                    or data["design"] is None):
                error = "Keep requires an unexhausted hypothesis, real draft and a testable next action"
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
        data["next_actions"] = list(result.next_actions)
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
            seed = data["working_draft"] or data["champion"]
            context = coder_context(DesignRef.from_dict(data["design"]), data["prepared"]["identity"], number, seed["identity"],
                                    next_actions=data.get("next_actions", ()))
            binding = {"episode": number, "operation_id": operation_id, "window": window["id"],
                "design": data["design"], "prepared_id": data["prepared"]["identity"],
                "seed": seed, "context": context, "context_id": digest(context), "hypothesis": data["hypothesis"]}
            data["binding"] = binding
            data["counts"]["episodes"] = number
            self.state.save()  # Before native admission and seed installation.
        seed = self.state.read_source(binding["seed"], owner=campaign.candidate_workspace)
        def qualify(**native):
            source = campaign.candidate_workspace.read_source(native["worktree"].path,
                revision=native["candidate_commit"])
            identity = campaign.candidate_workspace.source_identity(source)
            if self.candidate_qualification is not None:
                return self.candidate_qualification(binding=dict(binding), source=source,
                    source_identity=identity, **native)
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
                    "seed_files": dict(original.files)})
            if result.get("binding", {}).get("source_identity") != identity:
                raise RecoveryRequired("Qualification receipt does not bind the sealed native source")
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
        progress = self._progress(binding, attempt, fresh, hypothesis=data["hypotheses"][binding["hypothesis"]])
        data["champion"], data["tested_candidate"], data["working_draft"] = champion, tested, draft
        data["episodes"][str(episode)] = {"episode": episode, "binding": binding,
            "status": attempt["status"], "accepted": attempt["accepted"], "violation": attempt.get("violation"),
            "tokens": attempt.get("tokens"), "telemetry": attempt.get("telemetry"),
            "draft": draft, "sealed_candidate": sealed, "tested_candidate": tested,
            "champion": champion, "records": records, "measurements": measurements[-8:], "progress": progress}
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

    def _progress(self, binding, attempt, fresh, *, hypothesis=None):
        if hypothesis is None:
            hypothesis = self.state.data["hypotheses"][binding["hypothesis"]]
        qualified = attempt.get("candidate_qualification") or {}
        if qualified.get("qualified") is not True or qualified.get("substantive_change") is not True:
            return []
        progress = []
        for record, source in fresh:
            identity = source["identity"]
            if identity == hypothesis["seed"]["identity"]:
                continue
            evaluation = record.evaluation
            measurements = evaluation.measurements if evaluation else ()
            complete = (evaluation is not None and evaluation.gate == "PASS" and not evaluation.error and measurements
                and all(item.completeness == "complete" and item.validity == "valid" and item.comparability == "comparable"
                        and item.environment_identity and item.identity.parameters.get("workload_scope") == "full-task"
                        and item.identity.parameters.get("mode") == "full"
                        and len(item.identity.parameters.get("workload_digest", "")) == 64 for item in measurements))
            if not complete:
                continue
            # Native completion qualifies one exact sealed source; another
            # measured trial does not inherit its fidelity judgment.
            if qualified.get("source_identity") != identity:
                continue
            if identity in hypothesis["seen_sources"]:
                continue  # Remeasurement/replay of old bytes is not new progress.
            hypothesis["seen_sources"].append(identity)
            for milestone in ("first-compiled", "first-correct", "first-comparable-performance"):
                if milestone not in hypothesis["milestones"]:
                    hypothesis["milestones"].append(milestone)
                    progress.append(milestone)
            latency = evaluation.candidate_latency_us
            if type(latency) not in (float, int) or not math.isfinite(latency) or latency <= 0:
                continue
            # Same native measurement protocol, workload, dependency policy,
            # repeats and environment; source coordinates intentionally vary.
            signatures = []
            for item in measurements:
                value = item.identity.as_dict()
                # Native input_digest includes the candidate payload. The
                # separate workload_digest is the source-independent binding,
                # as in the original Experiment performance-reuse query.
                value.pop("input_digest")
                value["parameters"] = {key: val for key, val in value["parameters"].items()
                    if not key.startswith("candidate_source") and key not in {"correctness_digest"}}
                value["points"] = [{k: v for k, v in point.items() if k != "revision"} for point in value["points"]]
                signatures.append({"identity": value, "environment": item.environment_identity})
            key = digest(signatures)
            best = hypothesis["best_legal"]
            threshold = float(getattr(self._campaign, "min_improvement_pct", 0))
            if best and best["comparison_key"] == key and latency < best["latency_us"] * (1 - threshold / 100):
                progress.append("comparable-performance-improved")
            if best is None or best["comparison_key"] == key and latency < best["latency_us"]:
                hypothesis["best_legal"] = {"latency_us": latency, "comparison_key": key,
                    "experiment": _record_ref(record), "source": source}
        return progress

    def _close_window(self, *, status="settled", stop_reason=None):
        data = self.state.data
        window = data["window"]
        hypothesis = data["hypotheses"][window["hypothesis"]]
        hypothesis["stagnant"] = 0 if window["progress"] else hypothesis["stagnant"] + 1
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
