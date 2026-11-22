"""Schema consumes real native codecs with retained synthetic CPU job results."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from aka.contracts.execution import ExecutionFacts, JobResult
from aka.contracts.evaluation import EvaluationConfiguration
from aka.contracts.experiment import ExperimentRecordError
from aka.contracts.workspace import RevisionContent
from aka.evaluation.gateway_command import GatewayCommandEvaluator
from aka.experiment.file_records import FileExperimentRecords
from aka.experiment.runner import LegacyExperiment
from aka.execution.legacy.executor import LegacyExecutor
from aka.task.candidate_workspace.provider import GitCandidateWorkspace
from aka.workflow.schema.progress import needs_new_hypothesis
from aka.workflow.schema.workflow import SchemaWorkflow
from aka.workflow.schema.run_state import RecoveryRequired, RunLimits
from test_native_environment_compat import request, raw_job, snapshot
from test_schema_workflow import controller, native, state_at


SEED = b"# original synthetic design seed\n"


@pytest.fixture
def run(tmp_path):
    environment = json.loads((Path(__file__).parent / "fixtures/agate_b41_environment.json").read_text())
    state = state_at(tmp_path)
    owner = GitCandidateWorkspace()
    experiments = {}
    records = SimpleNamespace(
        read_source=lambda artifact: experiments[artifact.experiment].read_source(artifact),
        read_artifact=lambda artifact: experiments[artifact.experiment].read_artifact(artifact),
        command_result=lambda ref: experiments[ref].command_result(ref),
        iter_records=lambda: (record for experiment in experiments.values() for record in experiment.iter_records()))
    _, task = request(tmp_path)
    driver = SchemaWorkflow.__new__(SchemaWorkflow)
    driver.state = state
    driver._campaign = SimpleNamespace(candidate_workspace=owner, problem=SimpleNamespace(evaluation=task),
        sandbox_hardware="L20D", min_improvement_pct=0, experiment_record_reader=lambda: nullcontext(records))
    binding = {"hypothesis": "h", "prepared_id": "task", "design": {"identity": "design"}}
    def measured(label, latency=20, *, mode="full", abba=False, baseline=SEED,
                 baseline_latency=25, comparable=False, damage=None):
        folder = tmp_path / ("record-" + str(len(experiments)))
        command, local_task = request(folder, mode=mode, abba=abba)
        wire = json.loads(command.submission)
        wire["request"]["spec"] = {"target_hardware": ["L20D"], "languages": ["triton"]}
        wire["request"]["candidate"] += "# " + label + "\n"
        if abba:
            wire["request"]["abba"]["baseline"] = baseline.decode()
        if damage == "task":
            wire["request"]["reference"]["input_py"] += "# different task\n"
        elif damage == "shape_input":
            wire["request"]["reference"]["shapes"]["2"] = {"N": 1}
        elif damage == "subset":
            del wire["request"]["reference"]["shapes"]["2"]
        elif damage == "hardware":
            wire["request"]["spec"]["target_hardware"] = ["different-device"]
        command = replace(command, submission=json.dumps(wire).encode())
        job = raw_job(mode=mode, abba=abba)
        results = ([(row["result"], latency if row["revision"] == "candidate" else baseline_latency)
                    for row in job["result"]["abba"]["sdk_results"][0]["abba"]["runs"]]
                   if abba else [(job["result"], latency)])
        for result, timing in results:
            for shape in result.get("performance", {}).get("shapes", {}).values():
                shape["samples"][0]["end_to_end_time_ms"] = timing / 1000
        if damage == "missing_shape":
            del results[0][0]["passed"]["correctness"]["2"]
        elif damage == "missing_run":
            job["result"]["abba"]["sdk_results"][0]["abba"]["runs"].pop()
        elif damage == "wrong_order":
            job["result"]["abba"]["sdk_results"][0]["abba"]["runs"][1]["index"] = 2
        elif damage == "failed_correctness":
            results[0][0]["passed"]["correctness"]["2"]["status"] = "failed"
        env = snapshot(environment)
        receipt = JobResult(0, stdout=json.dumps(job), transport_status="succeeded",
            facts=ExecutionFacts(disposition="terminal"), environment_snapshot=env,
            environment_identity="cpu-fixture-worker" if comparable else None)
        evaluator = GatewayCommandEvaluator(execution_capabilities=LegacyExecutor.capabilities,
            evaluation_input=local_task, shape_batch_size=3)
        experiment = LegacyExperiment(SimpleNamespace(capabilities=LegacyExecutor.capabilities, run=lambda _: receipt),
            records=FileExperimentRecords(folder / "records", "schema-progress-cpu-" + str(len(experiments))), command_evaluator=evaluator)
        record = experiment.run_recorded(command, request_id="evaluation")
        experiments[record.ref] = experiment
        # Each fake owner has a distinct namespace in actual storage, even when
        # the same native request name is used.
        source = experiment.read_source(next(a for a in record.artifacts if a.role == "candidate-source"))
        source_ref = state.retain_source(source, owner=owner, origin={"role": "tested"})
        return record, source_ref
    def progress(items, *, accepted=False, **qualification):
        source = items[0][1]["identity"]
        attempt = {"accepted": accepted, "candidate_qualification": {"qualified": True,
            "substantive_change": True, "source_identity": source, **qualification}}
        assessment = {}
        events = driver._progress(binding, attempt, items, assessment=assessment)
        return events, assessment
    with state.owned():
        state.data["prepared"] = {"public_task": "synthetic current task", "hardware": "L20D"}
        seed = state.retain_source(RevisionContent({"kernel.py": SEED}), owner=owner, origin={"role": "seed"})
        state.data["hypotheses"]["h"] = driver._new_hypothesis(seed)
        yield SimpleNamespace(driver=driver, state=state, measured=measured, progress=progress,
            hypothesis=state.data["hypotheses"]["h"], records=records, owner=owner, task=task)


def test_standard_attempt_fallback_counts_once_without_cross_record_identity(run):
    item = run.measured("first")
    assert not item[0].evaluation.measurements
    assessment = item[0].attempts[0].assessment
    assert assessment.completeness == "complete" and assessment.comparability == "unknown"
    events, status = run.progress([item])
    assert events == ["first-compiled", "first-correct", "first-qualified-full-measurement"]
    assert status["status"] == "progress"
    assert run.hypothesis["best_legal"] is None
    before = deepcopy(run.state.data)
    assert run.progress([item])[0] == []
    assert run.progress([run.measured("first", 1)])[0] == []
    assert run.state.data == before


def test_correctness_only_never_becomes_performance(run):
    events, _ = run.progress([run.measured("correct", mode="correctness_only")])
    assert events == ["first-compiled", "first-correct"]
    assert run.hypothesis["best_legal"] is None
    assert "native_frontier" not in run.hypothesis


def test_complete_same_allocation_abba_uses_native_schedule(run):
    item = run.measured("abba", 20, abba=True)
    measurement = item[0].attempts[0].assessment
    assert measurement.identity.parameters["workload_scope"] == "subset-or-custom"
    assert measurement.environment_identity is None
    events, _ = run.progress([item])
    assert "first-qualified-full-measurement" in events
    assert "native-comparison-improved" in events
    assert "first-comparable-performance" not in events


def test_mixed_records_for_one_sealed_source_do_not_hide_abba(run):
    standard = run.measured("same", 20)
    abba = run.measured("same", 19, abba=True)
    events, _ = run.progress([standard, abba])
    assert events.count("first-correct") == 1
    assert "native-comparison-improved" in events


def test_unknown_is_visible_and_does_not_claim_improvement(run):
    run.progress([run.measured("first", 20)])
    events, assessment = run.progress([run.measured("second", 10)])
    assert events == []
    assert assessment["status"] == "unknown"
    assert assessment["reasons"] == ["cross-record-comparison-unavailable"]
    assert "native ABBA" in assessment["next_evidence"]
    assert assessment["comparison_target"] == "kernel-" + run.hypothesis["native_frontier"]["source"]["identity"]


def test_unpromoted_design_improvement_uses_history_best_not_last_result(run):
    # Every candidate can be slower than an external 5us champion. This design
    # still improves 20 -> 15; a 30 -> 18 rebound is not a new best.
    events = []
    for label, latency in (("a", 20), ("b", 15), ("c", 30), ("d", 18)):
        events.append(run.progress([run.measured(label, latency, comparable=True)])[0])
    assert "first-comparable-performance" in events[0]
    assert events[1] == ["comparable-performance-improved"]
    assert events[2:] == [[], []]
    assert run.hypothesis["best_legal"]["latency_us"] == pytest.approx(15)


def test_native_comparison_must_use_retained_best_source(run):
    first = run.measured("a", 20)
    run.progress([first])
    source = run.state.read_source(first[1], owner=run.owner).files["kernel.py"]
    events, _ = run.progress([run.measured("b", 15, abba=True, baseline=source, baseline_latency=20)])
    assert "native-comparison-improved" in events
    # Comparing again to a discarded slower source cannot refresh progress.
    events, assessment = run.progress([run.measured("c", 18, abba=True, baseline=source, baseline_latency=20)])
    assert events == [] and assessment["status"] == "unknown"


def test_unpromoted_target_reaches_coder_and_native_baseline_request(run, tmp_path):
    from aka.evaluation.tool_requests import request_from_argv
    from aka.workflow.light.measurements import SharedMeasurements
    from aka.workflow.light.host.workspace import snapshot as snapshot_workspace
    from aka.workflow.schema.materials import publish_design, snapshot_inputs
    first = run.measured("unpromoted")
    run.progress([first], accepted=False)
    target = "kernel-" + first[1]["identity"]
    materials = snapshot_inputs(tmp_path / "materials", ())
    design = publish_design(tmp_path / "design", Path(__file__).parent / "fixtures/schema_language/minimal",
        version=1, prepared_id="task", operation_id="cpu-design", materials_snapshot=materials)
    run.state.data.update(design=design.as_dict(), hypothesis="h", working_draft=first[1])
    run.state.data["prepared"]["identity"] = "task"
    run.driver._campaign.max_iters = 0
    def capture_engine(**kwargs):
        raise RuntimeError("captured native Coder context")
    run.driver._campaign.create_engine = capture_engine
    with pytest.raises(RuntimeError, match="captured native Coder context"):
        run.driver._implement(run.driver._campaign)
    context = run.state.data["binding"]["context"]
    commands = [shlex.split(line)[2:] for line in context.splitlines() if line.startswith("python3 tools/sandbox.py ")]
    assert len(commands) == 2 and target in commands[0]
    # Exercise the actual query API, not just the Schema progress classifier.
    # The original record has no accepted/champion status or environment identity.
    queries = SharedMeasurements(factory=SimpleNamespace(open=lambda **kw: nullcontext(run.records)),
        executor=None, evaluator=None, namespace_workspace=tmp_path, evaluation_input=run.task,
        workspace_provider=run.owner, command_evaluator=None)
    workspace = tmp_path / "coder"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("class Model:\n    pass\n# next candidate\n")
    configuration = EvaluationConfiguration(hardware="L20D", repeats=2, per_run_timeout=600)
    read = request_from_argv(commands[0], workspace=workspace, evaluation_input=run.task,
        configuration=configuration, workspace_provider=run.owner)
    assert read.request is None
    assert json.loads(queries.query(read.args, workspace)["stdout"])["kernel_id"] == target
    baseline = workspace / read.args.output_path
    original = run.state.read_source(first[1], owner=run.owner).files["kernel.py"]
    assert baseline.read_bytes() == original
    # The Runtime snapshots scratch before using the same native converter.
    staged = tmp_path / "staged"
    snapshot_workspace(workspace, staged)
    comparison = request_from_argv(commands[1], workspace=staged, evaluation_input=run.task,
        configuration=configuration, workspace_provider=run.owner)
    assert comparison.request.operation == "same_allocation_abba"
    prepared = GatewayCommandEvaluator(execution_capabilities=LegacyExecutor.capabilities,
        evaluation_input=run.task, shape_batch_size=3).prepare(comparison.request)
    assert all(identity.parameters["baseline_source_sha256"] == hashlib.sha256(original).hexdigest()
               for identity in prepared.measurement_identities)
    assert {point.shape_id for identity in prepared.measurement_identities for point in identity.points} == {"0", "1", "2"}


def test_native_promotion_is_progress_even_when_champion_differs_from_design_best(run):
    run.progress([run.measured("slow-design", 30)])
    events, status = run.progress([run.measured("promoted", 15, abba=True, baseline_latency=20)], accepted=True)
    assert events == ["native-promotion"]
    assert status["status"] == "progress"


def test_missing_prepared_evaluation_repetition_is_not_full_measurement(run):
    record, source = run.measured("missing-repeat")
    identity = record.steps[0].measurement_identity
    identity = replace(identity, parameters={**identity.parameters, "measurement_repetitions": 3})
    record = replace(record, steps=(replace(record.steps[0], measurement_identity=identity),),
        attempts=(replace(record.attempts[0], assessment=replace(record.attempts[0].assessment, identity=identity)),))
    assert run.progress([(record, source)])[0] == []


@pytest.mark.parametrize("damage", ["task", "shape_input", "subset", "missing_shape", "failed_correctness", "hardware"])
@pytest.mark.parametrize("abba", [False, True])
def test_task_shape_and_native_correctness_must_match(run, damage, abba):
    events, _ = run.progress([run.measured("bad", abba=abba, damage=damage)])
    assert events == []
    assert run.hypothesis["seen_sources"] == []


@pytest.mark.parametrize("damage", ["missing_run", "wrong_order"])
def test_incomplete_native_schedule_is_not_progress(run, damage):
    assert run.progress([run.measured("bad", abba=True, damage=damage)])[0] == []


def test_wrong_source_and_qualification_are_not_progress(run):
    item = run.measured("valid")
    assert run.progress([item], source_identity="different")[0] == []
    assert run.progress([item], binding={"source_identity": item[1]["identity"],
        "prepared_id": "different-task", "design_identity": "design"})[0] == []
    assert run.progress([item], qualified=False)[0] == []


def test_corrupt_source_binding_fails_closed_without_consuming_progress(run):
    record, source = run.measured("wrong-source")
    identity = record.steps[0].measurement_identity
    identity = replace(identity, parameters={**identity.parameters, "candidate_source_sha256": "f" * 64})
    record = replace(record, steps=(replace(record.steps[0], measurement_identity=identity),),
        attempts=(replace(record.attempts[0], assessment=replace(record.attempts[0].assessment, identity=identity)),))
    with pytest.raises(ExperimentRecordError, match="source_binding_mismatch"):
        run.progress([(record, source)])
    assert run.hypothesis["seen_sources"] == []


@pytest.mark.parametrize("field", ["design_identity", "prepared_id", "public_task_sha256", "target_hardware_sha256"])
def test_independent_qualification_binds_frozen_design_task_and_hardware(run, field):
    item = run.measured("reviewed")
    receipt = {"design_identity": "design", "prepared_id": "task", "source_identity": item[1]["identity"],
        "public_task_sha256": hashlib.sha256(b"synthetic current task").hexdigest(),
        "target_hardware_sha256": hashlib.sha256(b"L20D").hexdigest()}
    receipt[field] = "different"
    assert run.progress([item], binding=receipt)[0] == []


def test_uncertain_windows_do_not_count_stagnation_or_allow_unlimited_keep(run):
    state = run.state
    state.data["hypothesis"] = "h"
    run.hypothesis["stagnant"] = 1
    for episode in (1, 2):
        state.data["episodes"][str(episode)] = {"progress_assessment": {"status": "unknown"}}
        state.data["window"] = {"hypothesis": "h", "episodes": [episode], "progress": []}
        run.driver._close_window()
        assert run.hypothesis["stagnant"] == 1
        assert state.data["windows"][-1]["progress_status"] == "unknown"
        assert needs_new_hypothesis(run.hypothesis) is (episode == 2)


def test_unknown_window_cannot_repeat_the_same_planner_action_forever(tmp_path, monkeypatch):
    campaign = native(tmp_path)
    driver, planner = controller(tmp_path, campaign,
        limits=RunLimits(continuous=True, max_episodes=8, planner_attempts=2, retry_delays=(0,)))
    original = planner._plan
    def repeat_action(kind, request):
        return replace(original(kind, request), next_actions=("Implement rewritten arithmetic",))
    monkeypatch.setattr(planner, "_plan", repeat_action)
    def unknown(*args, assessment, **kwargs):
        assessment.update(status="unknown", reasons=["synthetic missing comparison evidence"])
        return []
    monkeypatch.setattr(driver, "_progress", unknown)
    with pytest.raises(RecoveryRequired, match="Planner repair/replan attempts exhausted"):
        driver.run(campaign)
    hypothesis = driver.state.data["hypotheses"]["h1"]
    assert hypothesis["stagnant"] == 0 and hypothesis["uncertain_windows"] == 1
    assert driver.state.data["counts"]["settled_episodes"] == 4
    assert any("untried evidence-gathering action" in item["message"] for item in driver.state.data["diagnostics"])
