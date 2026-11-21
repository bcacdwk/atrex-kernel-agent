"""CPU-only native receipts: correctness does not imply cross-record comparability."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aka.contracts.content import WorkspaceFile
from aka.contracts.execution import CommandSpec, ExecutionFacts, JobResult, UnsupportedExecution
from aka.contracts.execution_environment import EnvironmentQuery
from aka.contracts.problem import EvaluationInput
from aka.evaluation.gateway_command import GatewayCommandEvaluator
from aka.execution.environment_query import _snapshot, match_environment
from aka.execution.gateway.executor import GatewayExecutor
from aka.execution.legacy.executor import LegacyExecutor
from aka.experiment.file_records import FileExperimentRecords
from aka.experiment.runner import LegacyExperiment
from aka.task.candidate_workspace.provider import GitCandidateWorkspace
from aka.workflow.light.measurements import SharedMeasurements


@pytest.fixture
def service_environment():
    # These are recorded service fields, not a claimed worker/image identity.
    return json.loads((Path(__file__).parent / "fixtures/agate_b41_environment.json").read_text())


def snapshot(fields, backend="legacy"):
    return _snapshot(backend, "default-gateway", EnvironmentQuery("L20D"),
                     fields["detail"], fields["capabilities"], [], LegacyExecutor.capabilities.operations)


def test_current_service_environment_preserves_unknown_identity(service_environment):
    observed = snapshot(service_environment)
    assert observed.identity is None
    assert observed.observed["toolchain"]["triton"] == "3.7.1"
    assert not {"worker", "worker_id", "image", "image_digest"}.intersection(observed.observed)
    ordinary = match_environment(EnvironmentQuery("L20D", "run", ("triton",)), observed)
    abba = match_environment(EnvironmentQuery("L20D", "same_allocation_abba", ("triton",)), observed)
    assert ordinary.support == "supported"
    assert abba.dimensions["operation"] == "unknown"
    assert ordinary.availability == abba.availability == "unknown"


def request(workspace, *, mode="full", abba=False, cases=6):
    source = "class Model:\n    pass\n"  # Synthetic CPU fixture; never executed on GPU.
    shapes = {str(i): {"N": n} for i, n in enumerate((1024, 65536, 1048576))}
    reference = {"reference_py": source, "input_py": "# synthetic CPU fixture\n", "shapes": shapes,
        "metadata": {"perf_gpu": "L20D", "shapes": {
            sid: {"production_performance": {"performance_us": 20}} for sid in shapes}}}
    wire = {"candidate": source, "reference": reference, "mode": mode,
        "spec": {"gpu": "L20D", "languages": ["triton"], "num_gpus": 1},
        "options": {"num_correctness_cases": cases}, "deps_mode": "preinstalled",
        "runner_overrides": {"perf_timeout_s": 120}}
    if abba:
        wire["abba"] = {"baseline": source + "# baseline\n", "repeats": 1}
    files = tuple(WorkspaceFile(name, text=json.dumps(reference[key]) if name.endswith(".json") else reference[key])
        for name, key in (("reference.py", "reference_py"), ("input.py", "input_py"),
                          ("shapes.json", "shapes"), ("metadata.json", "metadata")))
    task = EvaluationInput("legacy-native-v1", workspace, files, case_ids=tuple(shapes))
    command = CommandSpec(workspace, ("python3", "test_kernel.py"), hardware="L20D", timeout=120,
        operation="same_allocation_abba" if abba else "run",
        submission=json.dumps({"schema_version": 1, "kind": "run", "request": wire}).encode())
    return command, task


def raw_evaluation(*, correctness_only=False):
    shapes = ("0", "1", "2")
    value = {"error": None, "passed": {
        stage: {sid: {"status": "passed"} for sid in shapes}
        for stage in ("compile", "correctness", "performance")},
        "correctness": {"shapes": {sid: {"cases": [{"outputs": []} for _ in range(6)]} for sid in shapes}}}
    if not correctness_only:
        value["performance"] = {"shapes": {
            sid: {"samples": [{"end_to_end_time_ms": 0.01}]} for sid in shapes}}
    return value


def raw_job(*, mode="full", abba=False):
    if not abba:
        return {"job_id": "cpu-synthetic-eval", "status": "succeeded", "result": raw_evaluation(correctness_only=mode == "correctness_only")}
    rows = []
    for index, (revision, label, repeat) in enumerate((
            ("baseline", "A", 0), ("candidate", "B", 0), ("candidate", "B", 1), ("baseline", "A", 1))):
        rows.append(dict(index=index, revision=revision, label=label, repeat=repeat, result=raw_evaluation()))
    return {"job_id": "cpu-synthetic-abba", "status": "succeeded", "result": {"abba": {"valid": True, "sdk_results": [
        {"passed": {"abba": {"status": "passed"}}, "abba": {"runs": rows}}]}}}


def record_job(tmp_path, service_environment, *, mode="full", abba=False, job=None, cases=6):
    command, task = request(tmp_path, mode=mode, abba=abba, cases=cases)
    env = snapshot(service_environment)
    receipt = JobResult(0, stdout=json.dumps(job or raw_job(mode=mode, abba=abba)),
        transport_status="succeeded", facts=ExecutionFacts(disposition="terminal"),
        environment_snapshot=env, environment_identity=env.identity)
    executor = SimpleNamespace(capabilities=LegacyExecutor.capabilities, run=lambda command: receipt)
    evaluator = GatewayCommandEvaluator(execution_capabilities=executor.capabilities,
        evaluation_input=task, shape_batch_size=3)
    experiment = LegacyExperiment(executor, records=FileExperimentRecords(tmp_path / "records", "cpu-native"),
        command_evaluator=evaluator)
    record = experiment.run_recorded(command, request_id="evaluation")
    view = SharedMeasurements(factory=None, executor=None, evaluator=None, namespace_workspace=tmp_path,
        evaluation_input=task, workspace_provider=GitCandidateWorkspace(), command_evaluator=evaluator)
    return experiment, record, view


@pytest.mark.parametrize("mode", ["full", "correctness_only"])
def test_native_standard_and_six_case_truth_without_cross_record_reuse(tmp_path, service_environment, mode):
    experiment, record, view = record_job(tmp_path, service_environment, mode=mode)
    evidence = view.project(experiment, record)
    assert evidence["execution_status"] == "completed"
    assert evidence["typed_six_case_correctness"] is True
    assert evidence["standard_full_evaluation"] is (mode == "full")
    assert evidence["result"]["all_pass"] is True
    assert evidence["result"]["environment_identity"] is None
    assert evidence["result"]["comparability"] == "unknown"
    assert "qualification" not in evidence["result"]
    response = view._response(experiment, record)
    assert response["agent_result"]["result"]["correct"] is True
    assert response["agent_result"]["result"]["comparability"] == "unknown"
    assert response["agent_result"]["result"]["environment_identity"] is None
    identities = view._identities(record)
    assert all(item.comparability == "unknown" and item.environment_identity is None
               for item in view._measurements(record))
    assert not experiment.reusable(record.ref, identities)
    for environment in (None, "unrelated-known-worker"):
        assert experiment.correctness_reuse(identities, environment_identity=environment) is None
        assert experiment.shape_minima(identities, environment_identity=environment) == ()
    if mode == "full":
        from aka.workflow.light.host.journal import _passing_evaluate
        assert _passing_evaluate(evidence, evidence["kernel_artifact_digest"])
    from aka.workflow.light.candidate_gate import CandidateGate, CandidateValidation
    owner = GitCandidateWorkspace()
    context = CandidateValidation(atrex_bench=True, production=False,
        measure=lambda *args: response, read_record=lambda ref: evidence, ensure_live=lambda: None,
        source_identity=owner.source_identity)
    source = view.candidate_source(experiment, record)
    assert CandidateGate().validate_candidate(context, source)["correctness_cases"] == 6


def test_one_case_evaluation_does_not_become_six_case_evidence(tmp_path, service_environment):
    from aka.workflow.light.candidate_gate import CandidateGate, CandidateValidation
    from aka.workflow.light.host.errors import RuntimeStateError
    job = raw_job(mode="correctness_only")
    for shape in job["result"]["correctness"]["shapes"].values():
        shape["cases"] = shape["cases"][:1]
    experiment, record, view = record_job(tmp_path, service_environment,
        mode="correctness_only", cases=1, job=job)
    evidence = view.project(experiment, record)
    assert evidence["result"]["all_pass"] is True
    assert evidence["typed_six_case_correctness"] is False
    context = CandidateValidation(atrex_bench=True, production=False,
        measure=lambda *args: view._response(experiment, record), read_record=lambda ref: evidence,
        ensure_live=lambda: None, source_identity=GitCandidateWorkspace().source_identity)
    with pytest.raises(RuntimeStateError, match="insufficient identity or coverage"):
        CandidateGate().validate_candidate(context, view.candidate_source(experiment, record))


def test_native_abba_uses_complete_sdk_schedule_without_environment_identity(tmp_path, service_environment):
    experiment, record, view = record_job(tmp_path, service_environment, abba=True)
    evidence = view.project(experiment, record)
    assert evidence["execution_status"] == "completed"
    assert evidence["result"]["complete"] is True
    assert evidence["result"]["correct"] is True
    assert evidence["result"]["comparison"] == {"method": "abba", "repeats": 2}
    assert evidence["result"]["environment_identity"] is None
    assert evidence["result"]["comparability"] == "unknown"
    assert not experiment.reusable(record.ref, view._identities(record))


@pytest.mark.parametrize("damage", ["missing_shape", "missing_time", "failed_correctness", "error"])
def test_native_result_does_not_invent_missing_or_failed_evidence(tmp_path, service_environment, damage):
    job = raw_job()
    if damage == "missing_shape":
        del job["result"]["passed"]["correctness"]["2"]
    elif damage == "missing_time":
        job["result"]["performance"]["shapes"]["2"]["samples"] = []
    elif damage == "failed_correctness":
        job["result"]["passed"]["correctness"]["2"]["status"] = "failed"
    else:
        job["result"]["error"] = "synthetic unavailable evaluator"
    experiment, record, view = record_job(tmp_path, service_environment, job=job)
    evidence = view.project(experiment, record)
    assert evidence["result"]["all_pass"] is not True
    from aka.workflow.light.host.journal import _passing_evaluate
    assert not _passing_evaluate(evidence, evidence["kernel_artifact_digest"])


@pytest.mark.parametrize("damage", ["missing_run", "wrong_order", "missing_time", "environment_rejected"])
def test_native_abba_incomplete_or_rejected_sdk_evidence_stays_unknown(tmp_path, service_environment, damage):
    job = raw_job(abba=True)
    block = job["result"]["abba"]["sdk_results"][0]
    if damage == "missing_run":
        block["abba"]["runs"].pop()
    elif damage == "wrong_order":
        block["abba"]["runs"][1]["index"] = 2
    elif damage == "missing_time":
        del block["abba"]["runs"][2]["result"]["performance"]["shapes"]["1"]
    else:
        block["passed"]["abba"] = {"status": "failed",
            "reason": "ABBA runs reported different GPU/runtime environments"}
    experiment, record, view = record_job(tmp_path, service_environment, abba=True, job=job)
    evidence = view.project(experiment, record)
    assert evidence["execution_status"] == "unknown"
    assert evidence["result"]["complete"] is False
    assert evidence["result"]["correct"] is None
    assert "native-abba-incomplete" in evidence["result"]["qualification"]["reasons"]


@pytest.mark.parametrize("damage", ["source_missing", "assessment_missing", "identity_mismatch", "unsettled", "non_native"])
def test_native_projection_keeps_original_source_and_terminal_guards(tmp_path, service_environment, damage):
    experiment, record, view = record_job(tmp_path, service_environment)
    if damage == "source_missing":
        record = replace(record, artifacts=())
    elif damage == "assessment_missing":
        record = replace(record, attempts=(replace(record.attempts[0], assessment=None),))
    elif damage in {"identity_mismatch", "non_native"}:
        identity = replace(record.steps[0].measurement_identity, method="legacy-eval/command")
        step = replace(record.steps[0], measurement_identity=identity)
        attempts = record.attempts
        if damage == "non_native":
            attempts = (replace(attempts[0], assessment=replace(attempts[0].assessment, identity=identity)),)
        record = replace(record, steps=(step,), attempts=attempts)
    else:
        record = replace(record, phase="running", lifecycle="running")
    evidence = view.project(experiment, record)
    assert evidence["result"]["correct"] is None
    assert evidence["execution_status"] == "unknown"


@pytest.mark.parametrize("provider", ["legacy", "gateway", "replay"])
@pytest.mark.parametrize("declaration", ["absent", "supported", "unsupported"])
def test_typed_agate_abba_admission_keeps_unknown_and_rejects_explicit_unsupported(
        tmp_path, monkeypatch, service_environment, provider, declaration):
    command, _ = request(tmp_path, abba=True)
    fields = deepcopy(service_environment)
    if declaration != "absent":
        fields["capabilities"]["operations"] = ["run"] + (["same_allocation_abba"] if declaration == "supported" else [])
    env = snapshot(fields)
    support = match_environment(EnvironmentQuery("L20D", "same_allocation_abba", ("triton",)), env)
    backend = SimpleNamespace(ssh="", check=lambda *args, **kwargs: support)
    calls = []
    def receipt(*args, **kwargs):
        calls.append(True)
        return JobResult(0)
    if provider in {"legacy", "replay"}:
        from aka.contracts.execution import ExecutionContext
        executor = LegacyExecutor().bind(ExecutionContext({}, "L20D", 120))
        monkeypatch.setattr(executor, "_run_sandbox", receipt)
    else:
        executor = GatewayExecutor().bind(SimpleNamespace(environment={}, filesystem_access=None))
        monkeypatch.setattr("aka.execution.outcomes.run_owned", receipt)
    monkeypatch.setattr(executor, "_environment_backend", lambda *args: backend)
    monkeypatch.setattr(executor, "_job_transport", lambda: SimpleNamespace(
        environment={}, endpoint="", profile="", replay=receipt))
    run = (lambda: executor.replay_job(command, idempotency_key="cpu")) if provider == "replay" else lambda: executor.run(command)
    if declaration == "unsupported":
        with pytest.raises(UnsupportedExecution, match="unsupported:operation"):
            run()
        assert calls == []
    else:
        result = run()
        assert calls == [True]
        assert result.environment_identity is None
        assert result.environment_support.dimensions["operation"] == ("unknown" if declaration == "absent" else "supported")


def test_ssh_abba_still_requires_affirmative_backend_support(tmp_path, monkeypatch, service_environment):
    from aka.contracts.execution import ExecutionContext
    command, _ = request(tmp_path, abba=True)
    executor = LegacyExecutor().bind(ExecutionContext({"ATREX_SANDBOX_SSH": "fixture-host"}, "L20D", 120))
    support = match_environment(EnvironmentQuery("L20D", "same_allocation_abba", ("triton",)), snapshot(service_environment))
    monkeypatch.setattr(executor, "_environment_backend", lambda: SimpleNamespace(check=lambda *args, **kwargs: support))
    monkeypatch.setattr(executor, "_run_sandbox", lambda *args, **kwargs: pytest.fail("must reject before transport"))
    with pytest.raises(UnsupportedExecution, match="unknown:operation"):
        executor.run(command)
