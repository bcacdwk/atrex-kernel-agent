"""Independent R5 audits of retained Schema inputs and native continuation.

All model/process/measurement data below are synthetic CPU fixtures. These
checks do not establish model quality, GPU correctness or campaign stability.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from aka.agent.provider import AgentOptions
from aka.contracts.experiment import ExperimentArtifact, ExperimentEvaluation, ExperimentRecord
from aka.contracts.experiment_identity import ExperimentRef
from aka.contracts.measurement import MeasurementAssessment, MeasurementIdentity, MeasurementPoint
from aka.contracts.problem import ProblemSnapshot
from aka.contracts.workspace import RevisionContent
from aka.workflow.schema.materials import collect_design, publish_design, snapshot_inputs
from aka.workflow.schema.planner import Planner, PlannerConfig
from aka.workflow.schema.run_state import RecoveryRequired, RunLimits
from aka.workflow.schema.workflow import ExistingStart
from aka.workflow.light.long_horizon.store import CampaignStore
from schema_cpu_fixture import BASELINE
from test_schema_planner import CpuAgent, response
from test_schema_workflow import FIXTURE, FakePlanner, ScopedFakeAgent, controller, native, state_at


def real_writer(root, campaign):
    """Real Planner and native Claude protocol with a synthetic process only."""
    calls = []
    def process(command, *, cwd, timeout, env):
        workspace = Path(cwd)
        calls.append(workspace)
        files = collect_design(FIXTURE)
        files["root.yaml"] = files["root.yaml"].replace(
            b"          y = x\n  Work_schedule:",
            b"          y = x + (2 - 2)\n  Work_schedule:")
        message = response(workspace, files=files)
        return json.dumps({"type": "result", "result": message,
            "usage": {"input_tokens": 4, "output_tokens": 3}}), "", 0, False
    agent = CpuAgent(AgentOptions(state_directory=root / "agent-state"), process_runner=process,
        environment={"HOME": str(root), "PATH": "/usr/bin:/bin",
                     "ANTHROPIC_MODEL": "qwen3.8-max[1m]"})
    campaign.agent = agent
    planner = Planner(agent, snapshot_inputs(root / "materials", ()),
                      config=PlannerConfig(timeout_seconds=7, max_repairs=0))
    return planner, calls


def paired_start(root, campaign, planner):
    ref = publish_design(root / "initial", FIXTURE, version=1,
        prepared_id="prepared-cpu", operation_id="paired-start",
        materials_snapshot=planner.materials_snapshot, mode="import")
    return ExistingStart(ref,
        campaign.candidate_workspace.source_identity(RevisionContent({"kernel.py": BASELINE})),
        "prepared-cpu", "L20D", ("synthetic-cpu-pair",))


@pytest.mark.parametrize("max_calls", [None, 3], ids=["continuous-call-allowance", "exact-finite-call-allowance"])
def test_optimize_recovery_replays_frozen_input_after_published_return_is_lost(tmp_path, monkeypatch, max_calls):
    campaign = native(tmp_path)
    planner, calls = real_writer(tmp_path, campaign)
    start = paired_start(tmp_path, campaign, planner)
    driver, _ = controller(tmp_path, campaign, planner=planner, initial=start,
                           limits=RunLimits(max_episodes=1, max_agent_calls=max_calls))
    original = planner.optimize
    failed = []
    def lost_return(request):
        result = original(request)
        if not failed:
            failed.append(True)
            raise OSError("synthetic lost published Planner return")
        return result
    monkeypatch.setattr(planner, "optimize", lost_return)
    try:
        with pytest.raises(OSError, match="lost published"):
            driver.run(campaign)
        # Diagnostics added by the failed controller turn must not mutate the
        # already-started operation's frozen feedback on a clean restart.
        driver.stop_event = SimpleNamespace(is_set=lambda: driver.state.data["phase"] == "implement")
        assert driver.run(campaign) == "interrupted: requested"
        assert len(calls) == 1
        assert driver.state.data["counts"]["agent_calls"] == 1
        assert driver.state.data["counts"]["tokens"] == 7
        assert driver.state.data["design_version"] == 2
    finally:
        planner.agent.close()


def test_keep_next_action_is_delivered_to_the_next_native_coder(tmp_path):
    campaign = native(tmp_path)
    class DirectedPlanner(FakePlanner):
        def optimize(self, request):
            return replace(super().optimize(request),
                next_actions=("Audit sentinel: test a 128-element vector load while retaining this graph.",))
    planner = DirectedPlanner(tmp_path, campaign.agent)
    driver, _ = controller(tmp_path, campaign, planner=planner, limits=RunLimits(max_episodes=5))
    assert driver.run(campaign) == "budget: episodes"
    assert "Audit sentinel: test a 128-element vector load" in campaign.agent.calls[4]["prompt"]
    assert "Audit sentinel" not in campaign.agent.calls[0]["prompt"]


def test_first_optimize_receives_the_actual_given_kernel_as_frozen_input(tmp_path):
    campaign = native(tmp_path)
    planner, calls = real_writer(tmp_path, campaign)
    start = paired_start(tmp_path, campaign, planner)
    driver, _ = controller(tmp_path, campaign, planner=planner, initial=start,
                           limits=RunLimits(max_episodes=1))
    driver.stop_event = SimpleNamespace(is_set=lambda: driver.state.data["phase"] == "implement")
    try:
        assert driver.run(campaign) == "interrupted: requested"
        inputs = calls[0] / "inputs"
        assert (inputs / "sources/champion/kernel.py").read_bytes() == BASELINE
        assert (inputs / "sources/working_draft/kernel.py").read_bytes() == BASELINE
        metadata = (inputs / "task/sources.json").read_text()
        assert hashlib.sha256(BASELINE).hexdigest() in metadata
        assert start.source_identity in (inputs / "task/feedback.json").read_text()
        assert not (inputs / "sources/tested_candidate").exists()
    finally:
        planner.agent.close()


def test_optimizer_gets_unpromoted_measurement_and_exact_distinct_sources(tmp_path):
    campaign = native(tmp_path)
    measured = RevisionContent({"kernel.py": b"# measured source differs from working draft\n"})
    ref = ExperimentRef("synthetic-cpu", "unpromoted-measurement")
    artifact = ExperimentArtifact(ref, "candidate", "candidate-source", "a" * 64, 30)
    identity = MeasurementIdentity("synthetic-eval",
        (MeasurementPoint("candidate", "candidate-revision", 0, "shape"),), "b" * 64,
        {"workload_scope": "full-task", "mode": "full", "workload_digest": "c" * 64},
        dependency_policy="fixed")
    assessment = MeasurementAssessment(identity, "complete", "valid", identity.points,
        comparability="comparable", environment_identity="synthetic-L20D")
    evaluation = ExperimentEvaluation("PASS", 17.125, 10, -71.25,
        runs=({"private_reference": "DO_NOT_PROJECT_RAW_EVALUATOR_RUNS"},),
        error="DO_NOT_PROJECT_RAW_EVALUATOR_ERROR", experiment_ref=ref, measurements=(assessment,))
    record = ExperimentRecord(ref, "unpromoted", "evaluation", "development", 1,
        "terminal", "completed", {}, "d" * 64, artifacts=(artifact,), evaluation=evaluation)
    campaign.experiment_source = lambda _: measured
    campaign._supervisor_runtime.read_experiment_records = lambda: (
        (record,) if CampaignStore.checkpoint(campaign.workspace)[0].episodes else ())
    driver, planner = controller(tmp_path, campaign, limits=RunLimits(max_episodes=5))
    assert driver.run(campaign) == "budget: episodes"
    request = planner.calls[-1][1]
    # Real Planner input projection is inspected even though this test uses the
    # fake writer to run native Episodes without an external model.
    reader = Planner(campaign.agent, planner.materials_snapshot)
    inputs = reader._inputs(request)
    assert inputs["sources/champion/kernel.py"] == BASELINE
    assert inputs["sources/tested_candidate/kernel.py"] == measured.files["kernel.py"]
    assert inputs["sources/working_draft/kernel.py"] != BASELINE
    feedback = inputs["task/feedback.json"].decode()
    assert "17.125" in feedback and "-71.25" in feedback
    assert "synthetic-L20D" in feedback and "complete" in feedback
    assert "DO_NOT_PROJECT_RAW_EVALUATOR" not in feedback
    summaries = [measurement for episode in json.loads(request.feedback[0]["text"])["episodes"]
                 for measurement in episode["measurements"]]
    measured_summary = next(row for row in summaries if row["experiment"] == ref.as_dict())
    assert measured_summary["candidate_source_identities"] == [
        campaign.candidate_workspace.source_identity(measured)]
    assert not any(row["progress"] for row in driver.state.data["episodes"].values())


@pytest.mark.parametrize("changed", ["hardware", "task"])
def test_finished_root_still_rejects_changed_native_identity(tmp_path, changed):
    campaign = native(tmp_path)
    driver, planner = controller(tmp_path, campaign, limits=RunLimits(max_episodes=1))
    assert driver.run(campaign) == "budget: episodes"
    rebuilt = native(tmp_path)
    planner.agent = rebuilt.agent
    if changed == "hardware":
        rebuilt.sandbox_hardware = "Blackwell"
    else:
        rebuilt.problem_provider.snapshot = lambda _: ProblemSnapshot(tmp_path, "different-task")
    resumed, _ = controller(tmp_path, rebuilt, planner=planner, limits=RunLimits(max_episodes=1))
    with pytest.raises(RecoveryRequired, match="identity|Native target"):
        resumed.run(rebuilt)
    assert not rebuilt.agent.calls and not rebuilt.evaluations


def test_agent_owner_recovery_charges_only_unobserved_active_time(tmp_path, monkeypatch):
    from aka.agent.execution.control import SessionControl
    from aka.core.active_time import ActiveBudget, budget_scope
    now = [100.0]
    monkeypatch.setattr("aka.core.active_time.time.monotonic", lambda: now[0])
    state = state_at(tmp_path, max_agent_active_seconds=40)
    with state.owned():
        state.action("prepare")
        state.data["counts"]["agent_active_seconds"] = 20
        control = SessionControl(suspend=False)
        permit = state.admit_agent(command=("synthetic",), workspace=tmp_path,
            timeout=10, environment={}, backend="claude", access=None, control=control)
        with budget_scope(ActiveBudget(40, used=20, checkpoint=state.checkpoint_active_time)):
            control.clock.start()
            now[0] += 3
            state.observe_agent(permit, event="output", value={"active_seconds": 3})
            control.clock.checkpoint(force=True)
        assert state.data["counts"]["agent_active_seconds"] == 23
        # Deliberately omit clock.close(), just as with a killed controller.
        # A new owner instance has none of the original process's controls.
    state = state_at(tmp_path, max_agent_active_seconds=40)
    with state.owned():
        state.reconcile(SimpleNamespace(), recover_agent=lambda *args: {
            "terminal": True, "active_seconds": 5, "basis": "synthetic original owner receipt"})
        assert state.data["counts"]["agent_active_seconds"] == 25
        assert state.data["counts"]["agent_calls"] == 1
        state.reconcile(SimpleNamespace())
        assert state.data["counts"]["agent_active_seconds"] == 25


def test_nested_agent_recovery_preserves_each_calls_checkpointed_share(tmp_path, monkeypatch):
    from aka.agent.execution.control import SessionControl
    from aka.core.active_time import ActiveBudget, budget_scope
    now = [100.0]
    monkeypatch.setattr("aka.core.active_time.time.monotonic", lambda: now[0])
    state = state_at(tmp_path, max_agent_active_seconds=60)
    with state.owned():
        state.action("prepare")
        state.data["counts"]["agent_active_seconds"] = 20
        with budget_scope(ActiveBudget(60, used=20, checkpoint=state.checkpoint_active_time)):
            controls, permits = [], []
            for number in range(2):
                control = SessionControl(suspend=False)
                permit = state.admit_agent(command=("synthetic", str(number)), workspace=tmp_path,
                    timeout=10, environment={}, backend="claude", access=None, control=control)
                controls.append(control)
                permits.append(permit)
                control.clock.start()
                now[0] += 3 if number == 0 else 2
                control.clock.pause()
            controls[0].clock.resume()
            now[0] += 1
            controls[0].clock.checkpoint(force=True)
        assert state.data["counts"]["agent_active_seconds"] == 26
    state = state_at(tmp_path, max_agent_active_seconds=60)
    final_seconds = {permits[0]: 6, permits[1]: 4}
    with state.owned():
        state.reconcile(SimpleNamespace(), recover_agent=lambda permit, call: {
            "terminal": True, "active_seconds": final_seconds[permit], "basis": "original owner"})
        assert state.data["counts"]["agent_active_seconds"] == 30
        assert state.data["counts"]["agent_calls"] == 2


def test_root_token_budget_blocks_second_coder_call_in_the_same_episode(tmp_path):
    campaign = native(tmp_path)
    class ResumingAgent(ScopedFakeAgent):
        def open_session(self, *args, **kwargs):
            session = super().open_session(*args, **kwargs)
            session.supports_resume = True
            original = session.invoke
            session.invoke = lambda *a, **kw: replace(original(*a, **kw), exit_status=0)
            return session
    campaign.agent = ResumingAgent()
    campaign.handoff_resumes = 3
    driver, _ = controller(tmp_path, campaign, limits=RunLimits(max_episodes=1, max_tokens=7))
    assert driver.run(campaign).startswith("budget:")
    assert len(campaign.agent.calls) == 1
    assert driver.state.data["counts"]["agent_calls"] == 1
    assert driver.state.data["counts"]["tokens"] == 7
    # Native archival and completed-root replay must not count that same
    # observed invocation a second time or refund the consumed allowance.
    assert driver.run(campaign).startswith("budget:")
    assert driver.state.data["counts"]["tokens"] == 7


def test_ready_public_delta_preserves_real_provider_approved_task_files(tmp_path):
    from aka.task.problem.legacy import LegacyProblemProvider
    from aka.workflow.schema.planner import PlannerRequest
    example = Path(__file__).resolve().parents[1] / "examples/schema/l20d/vector_add/benchmark"
    operator = tmp_path / "public-vector-add"
    operator.mkdir()
    # No evaluator runtime or GPU is needed to test the public content role.
    # Omitting shapes selects the real kernel-v1 adapter for this CPU probe.
    for name in ("reference.py", "input.py", "metadata.json"):
        shutil.copyfile(example / name, operator / name)
    requirements = tmp_path / "private-worker-requirements.txt"
    requirements.write_text("# DO_NOT_PROJECT_WORKER_REQUIREMENTS\n")
    provider = LegacyProblemProvider(worker_requirements=requirements)
    problem = provider.load(operator, snapshot_directory=tmp_path / "real-task-snapshot",
                            expose_evaluator_shapes=True)
    assert problem.evaluation.adapter == "legacy-kernel-v1"
    assert problem.agent.files

    campaign = native(tmp_path)
    # Only native V0/measurement is the established CPU fixture. Task loading,
    # snapshot/public roles and the Schema preparation projection are real.
    campaign.prepare_native()
    public = provider.content(problem, campaign.workspace).prepare_public()
    assert public.ready and public.files == ()
    campaign.problem = problem
    campaign.problem_provider = provider
    campaign.prepare_native = lambda: (public, provider.snapshot(problem))
    driver, planner = controller(tmp_path, campaign, limits=RunLimits(max_episodes=1))
    with driver.state.owned():
        driver._prepare(campaign)
        text = driver.state.data["prepared"]["public_task"]
        for item in problem.agent.files:
            assert f"## {item.path}\n" in text
            assert item.source.read_text() in text
        assert "DO_NOT_PROJECT_WORKER_REQUIREMENTS" not in text
        assert "private-worker-requirements" not in text
        assert driver.state.data["prepared"]["identity"] == provider.snapshot(problem).identity
        # The actual Planner validation must accept this task, without a model.
        reader = Planner(campaign.agent, planner.materials_snapshot)
        request = PlannerRequest(driver.state.data["prepared"]["identity"], "public-content", 1,
            tmp_path / "projection-operation", text, "L20D")
        assert reader._inputs(request)["task/public.md"] == text.encode()
