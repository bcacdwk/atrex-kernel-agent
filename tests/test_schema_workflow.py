"""CPU-only controller regression using the real Light engine and owned drafts."""
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aka.contracts.content import PublicInputContent, WorkspaceFile
from aka.contracts.agent import TokenUsage
from aka.contracts.execution import CommandSpec, ExecutionContext, ExecutionFacts, JobResult
from aka.contracts.execution_recovery import JobReference
from aka.contracts.experiment import ExperimentArtifact
from aka.contracts.experiment_identity import ExperimentRef
from aka.contracts.problem import ProblemSnapshot
from aka.contracts.workspace import RevisionContent
from aka.execution.legacy.executor import LegacyExecutor
from aka.workflow.light.long_horizon.store import CampaignStore
from aka.workflow.schema.materials import collect_design, publish_design, snapshot_inputs
from aka.workflow.schema.planner import PlannerResult
from aka.workflow.schema.run_state import BudgetDenied, RecoveryRequired, RunLimits, RunState
from aka.workflow.schema.workflow import ExistingStart, SchemaWorkflow, build_feedback, extract
from schema_cpu_fixture import BASELINE, CpuCampaign, FakeAgent


FIXTURE = Path(__file__).parent / "fixtures/schema_language/minimal"


class ScopedFakeAgent(FakeAgent):
    @contextmanager
    def invocation_scope(self, *, admit, observe):
        self.scope = (admit, observe)
        try:
            yield self
        finally:
            self.scope = None

    def open_session(self, agent_cli, workspace, **kwargs):
        session = super().open_session(agent_cli, workspace, **kwargs)
        original = session.invoke
        def invoke(*args, **kw):
            admit, observe = self.scope
            permit = admit(command=("fake-claude",), workspace=workspace, timeout=10,
                environment={}, backend="claude", access=None, control=None)
            try:
                result = original(*args, **kw)
            except BaseException as error:
                observe(permit, event="error", value=error)
                raise
            observe(permit, event="result", value={"active_seconds": .001,
                "result": (result.stdout, result.stderr, result.exit_status, result.timed_out)})
            return result
        session.invoke = invoke
        return session


class FakePlanner:
    def __init__(self, root, agent):
        self.materials_snapshot = snapshot_inputs(root / "materials", ())
        self.agent = agent
        self.calls = []
        self.observe_result = None
        self.results = {}
        self.bad = False

    def design(self, request):
        return self._plan("design", request)

    def optimize(self, request):
        return self._plan("optimize", request)

    def _plan(self, kind, request):
        if request.operation_id in self.results:
            return self.results[request.operation_id]
        self.calls.append((kind, request))
        if self.bad:
            result = PlannerResult("failed", None, request.operation_id, request.directory,
                diagnostics=({"code": "synthetic.bad_draft"},))
        elif request.current_design and not request.require_new:
            result = PlannerResult("keep", None, request.operation_id, request.directory,
                next_actions=("Make the next actual source edit and submit native evidence",))
        else:
            files = collect_design(FIXTURE)
            # Valid math rewrites are not gated by a few structural fields.
            files["root.yaml"] = files["root.yaml"].replace(b"y = x", f"y = x + ({request.version} - {request.version})".encode())
            ref = publish_design(request.directory / "design", files, version=request.version,
                prepared_id=request.prepared_id, operation_id=request.operation_id,
                materials_snapshot=self.materials_snapshot, parent=request.current_design,
                mode=kind)
            result = PlannerResult("replace", ref, request.operation_id, request.directory,
                hypothesis_key=f"h{request.version}", changed_execution_decisions=("Scalar math rewrite",),
                evidence=({"basis": "synthetic CPU fixture only"},), next_actions=("Implement rewritten arithmetic",))
        self.results[request.operation_id] = result
        return result


def native(root):
    campaign = CpuCampaign(root)
    campaign.agent = ScopedFakeAgent()
    campaign.max_iters = 0
    campaign.token_budget = 0
    campaign.sandbox_hardware = "L20D"
    campaign.problem_provider.snapshot = lambda problem: ProblemSnapshot(root, "prepared-cpu")
    original_content = campaign.problem_provider.content
    def content(*args, **kwargs):
        value = original_content(*args, **kwargs)
        value.prepare_public = lambda: PublicInputContent(True, (WorkspaceFile("task.md", text="Synthetic CPU softmax"),))
        return value
    campaign.problem_provider.content = content
    runtime = campaign._supervisor_runtime
    runtime.read_episode_journal = lambda episode: {}
    runtime.read_experiment_records = lambda: ()
    runtime.read_episode_draft = lambda episode: campaign.candidate_workspace.read_source(
        campaign.workspace / ".atrex_long_horizon/drafts" / f"e{episode:04d}/workspace")
    return campaign


def controller(root, campaign, *, limits=None, planner=None, initial=None, **kwargs):
    planner = planner or FakePlanner(root, campaign.agent)
    driver = SchemaWorkflow(root / "schema-root", planner=planner, executor=LegacyExecutor(),
        workspace=campaign.workspace, limits=limits or RunLimits(max_episodes=4), initial=initial,
        sleep=lambda _: None, **kwargs)
    return driver, planner


def test_native_windows_retain_draft_and_new_math_design_starts_at_champion(tmp_path, monkeypatch):
    monkeypatch.setattr("aka.workflow.schema.workflow.extract", lambda *a, **kw: pytest.fail("no-SOTA design called extract"))
    campaign = native(tmp_path)
    driver, planner = controller(tmp_path, campaign, limits=RunLimits(max_episodes=10))
    assert driver.run(campaign) == "budget: episodes"
    state = driver.state.data
    assert [kind for kind, _ in planner.calls] == ["design", "optimize", "optimize"]
    assert planner.calls[-1][1].require_new
    assert state["counts"]["settled_episodes"] == 10
    assert state["counts"]["agent_calls"] == 10
    assert state["counts"]["tokens"] == 70
    assert len(state["windows"]) == 2 and state["window"]["status"] == "partial"
    assert campaign.agent.calls[0]["received"] == BASELINE
    assert campaign.agent.calls[7]["received"].count(b"synthetic coder progress") == 7
    assert campaign.agent.calls[8]["received"] == BASELINE
    assert state["episodes"]["8"]["draft"]["identity"] != state["episodes"]["8"]["champion"]["identity"]
    assert state["episodes"]["8"]["tested_candidate"] is None
    assert state["hypotheses"]["h1"]["milestones"] == []
    assert campaign.setup_count == campaign.finish_count == 1
    assert CampaignStore.checkpoint(campaign.workspace)[0].episodes == 10
    assert len(build_feedback(driver.state)) <= 8000
    # Final result replay is read-only and cannot receive a fresh allowance.
    assert driver.run(campaign) == "budget: episodes"
    assert campaign.finish_count == 1 and len(campaign.agent.calls) == 10


def test_existing_pair_optimizes_before_first_native_episode_without_extract(tmp_path, monkeypatch):
    monkeypatch.setattr("aka.workflow.schema.workflow.extract", lambda *a, **kw: pytest.fail("supplied Schema called extract"))
    campaign = native(tmp_path)
    planner = FakePlanner(tmp_path, campaign.agent)
    ref = publish_design(tmp_path / "initial", FIXTURE, version=1, prepared_id="prepared-cpu",
        operation_id="paired-start", materials_snapshot=planner.materials_snapshot, mode="import")
    start = ExistingStart(ref, campaign.candidate_workspace.source_identity(RevisionContent({"kernel.py": BASELINE})),
        "prepared-cpu", "L20D", ("synthetic-pairing-receipt",))
    driver, _ = controller(tmp_path, campaign, planner=planner, initial=start, limits=RunLimits(max_episodes=1))
    assert driver.run(campaign) == "budget: episodes"
    assert [kind for kind, _ in planner.calls] == ["optimize"]
    assert len(campaign.agent.calls) == 1


def test_wrong_device_existing_pair_fails_before_planner(tmp_path):
    campaign = native(tmp_path)
    planner = FakePlanner(tmp_path, campaign.agent)
    ref = publish_design(tmp_path / "initial", FIXTURE, version=1, prepared_id="prepared-cpu",
        operation_id="paired-start", materials_snapshot=planner.materials_snapshot, mode="import")
    start = ExistingStart(ref, campaign.candidate_workspace.source_identity(RevisionContent({"kernel.py": BASELINE})),
        "prepared-cpu", "Blackwell", ("old-device",))
    driver, _ = controller(tmp_path, campaign, planner=planner, initial=start)
    with pytest.raises(ValueError, match="target device"):
        driver.run(campaign)
    assert not planner.calls and not campaign.agent.calls


def test_native_archive_then_root_failure_resumes_same_episode_once(tmp_path, monkeypatch):
    campaign = native(tmp_path)
    driver, planner = controller(tmp_path, campaign, limits=RunLimits(max_episodes=2))
    original = driver._settle
    failed = []
    def fail_once(*args):
        if not failed:
            failed.append(True)
            raise OSError("lost controller callback after native archive")
        return original(*args)
    monkeypatch.setattr(driver, "_settle", fail_once)
    with pytest.raises(OSError, match="lost controller"):
        driver.run(campaign)
    assert CampaignStore.checkpoint(campaign.workspace)[0].episodes == 1
    assert driver.state.data["binding"]["episode"] == 1
    rebuilt = native(tmp_path)
    planner.agent = rebuilt.agent
    resumed, _ = controller(tmp_path, rebuilt, planner=planner, limits=RunLimits(max_episodes=2))
    assert resumed.run(rebuilt) == "budget: episodes"
    assert len(campaign.agent.calls) + len(rebuilt.agent.calls) == 2
    assert resumed.state.data["counts"]["settled_episodes"] == 2
    assert resumed.state.data["counts"]["tokens"] == 14
    assert len(planner.calls) == 1


def test_measured_a_and_later_draft_b_remain_distinct_on_next_seed(tmp_path):
    campaign = native(tmp_path)
    measured = RevisionContent({"kernel.py": b"# synthetic measured source A\n"})
    ref = ExperimentRef("synthetic-cpu", "source-a")
    artifact = ExperimentArtifact(ref, "candidate", "candidate-source", "a" * 64, 30)
    record = SimpleNamespace(ref=ref, lifecycle="terminal", phase="completed", failure=None,
                             attempts=(), artifacts=(artifact,), evaluation=None)
    campaign.experiment_source = lambda selected: measured if selected == artifact else pytest.fail("wrong source owner")
    campaign._supervisor_runtime.read_experiment_records = lambda: ((record,) if CampaignStore.checkpoint(
        campaign.workspace)[0].episodes else ())
    driver, _ = controller(tmp_path, campaign, limits=RunLimits(max_episodes=2))
    assert driver.run(campaign) == "budget: episodes"
    first = driver.state.data["episodes"]["1"]
    assert first["tested_candidate"]["identity"] == campaign.candidate_workspace.source_identity(measured)
    assert first["tested_candidate"]["identity"] != first["draft"]["identity"]
    assert first["tested_candidate"]["origin"]["experiment"] == ref.as_dict()
    assert campaign.agent.calls[1]["received"] == campaign.agent.calls[0]["received"] + b"\n# synthetic coder progress\n"


def test_bad_planner_has_durable_bounded_repair_and_explicit_resume(tmp_path):
    campaign = native(tmp_path)
    driver, planner = controller(tmp_path, campaign)
    planner.bad = True
    with pytest.raises(RecoveryRequired, match="repair/replan"):
        driver.run(campaign)
    assert len(planner.calls) == 3 and not campaign.agent.calls
    with pytest.raises(RecoveryRequired, match="repair/replan"):
        driver.run(campaign)
    assert len(planner.calls) == 3
    planner.bad = False
    driver.resume_after_repair("Corrected synthetic writer response")
    assert driver.run(campaign) == "budget: episodes"
    assert driver.state.data["counts"]["episodes"] == 4


def test_agent_budget_denial_archives_native_episode_without_new_work(tmp_path):
    campaign = native(tmp_path)
    driver, _ = controller(tmp_path, campaign, limits=RunLimits(max_episodes=4, per_call_active_seconds=5))
    assert driver.run(campaign) == "Agent invocation exceeds configured single-call active-time limit"
    state, active = CampaignStore.checkpoint(campaign.workspace)
    assert active is None and state.episodes == 1
    assert state.attempts[0]["status"] == "budget_stopped"
    assert driver.state.data["counts"]["agent_calls"] == 0
    assert driver.state.data["episodes"]["1"]["draft"]["identity"] == driver.state.data["champion"]["identity"]
    assert not campaign.agent.calls


def test_window_allowance_returns_to_planner_without_finishing_continuous_chain(tmp_path):
    campaign = native(tmp_path)
    driver, planner = controller(tmp_path, campaign,
        limits=RunLimits(continuous=True, max_episodes=3, window_agent_calls=1))
    assert driver.run(campaign) == "budget: episodes"
    assert len(campaign.agent.calls) == 3 and campaign.finish_count == 1
    assert [window["status"] for window in driver.state.data["windows"]] == ["bounded", "bounded"]
    assert [kind for kind, _ in planner.calls] == ["design", "optimize", "optimize"]
    assert planner.calls[-1][1].require_new


def test_changed_native_target_refuses_resume_before_model_or_execution(tmp_path):
    campaign = native(tmp_path)
    stop = SimpleNamespace(is_set=lambda: bool(campaign.agent.calls))
    driver, planner = controller(tmp_path, campaign, limits=RunLimits(max_episodes=3), stop_event=stop)
    assert driver.run(campaign) == "interrupted: requested"
    rebuilt = native(tmp_path)
    rebuilt.sandbox_hardware = "Blackwell"
    planner.agent = rebuilt.agent
    resumed, _ = controller(tmp_path, rebuilt, planner=planner, limits=RunLimits(max_episodes=3))
    with pytest.raises(RecoveryRequired, match="Native target"):
        resumed.run(rebuilt)
    assert not rebuilt.agent.calls and not rebuilt.evaluations


def test_native_finalization_receipt_survives_lost_controller_return(tmp_path, monkeypatch):
    campaign = native(tmp_path)
    driver, planner = controller(tmp_path, campaign, limits=RunLimits(max_episodes=1))
    original = campaign.finish_native
    def lost_return(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("lost finalization return")
    monkeypatch.setattr(campaign, "finish_native", lost_return)
    with pytest.raises(OSError, match="lost finalization"):
        driver.run(campaign)
    assert driver.state.data["phase"] == "finalize" and campaign.finish_count == 1
    rebuilt = native(tmp_path)
    planner.agent = rebuilt.agent
    resumed, _ = controller(tmp_path, rebuilt, planner=planner, limits=RunLimits(max_episodes=1))
    assert resumed.run(rebuilt) == "budget: episodes"
    assert rebuilt.finish_count == 0 and not rebuilt.agent.calls


def test_infrastructure_retries_survive_restart_and_need_recorded_repair(tmp_path):
    from aka.workflow.light.orchestration.infrastructure_retry import InfrastructureUnavailable
    campaign = native(tmp_path)
    campaign.evaluation_failure = InfrastructureUnavailable("synthetic terminal preparation failure")
    driver, _ = controller(tmp_path, campaign, limits=RunLimits(max_episodes=1, retry_delays=(0,)))
    with pytest.raises(RecoveryRequired, match="retry allowance"):
        driver.run(campaign)
    count = len(campaign.evaluations)
    assert count == 2
    with pytest.raises(RecoveryRequired, match="retries exhausted"):
        driver.run(campaign)
    assert len(campaign.evaluations) == count
    campaign.evaluation_failure = None
    driver.resume_after_repair("Restored synthetic terminal evaluator")
    assert driver.run(campaign) == "budget: episodes"


def test_continuous_crosses_old_total_limits_and_only_explicit_finite_budget_stops(tmp_path):
    campaign = native(tmp_path)
    driver, _ = controller(tmp_path, campaign, limits=RunLimits(continuous=True, max_episodes=36))
    assert driver.run(campaign) == "budget: episodes"
    assert driver.state.data["counts"]["settled_episodes"] == 36
    assert driver.state.data["counts"]["windows"] == 9
    assert len(driver.state.data["windows"]) == 9
    assert campaign.setup_count == 1


def state_at(tmp_path, **limits):
    return RunState(tmp_path / "root", limits=RunLimits(continuous=True, **limits),
        workspace=tmp_path / "native", identity={"test": True})


def test_pending_dispatch_queries_original_owner_and_never_replays(tmp_path):
    state = state_at(tmp_path)
    ref = JobReference("legacy", "existing-job", endpoint="https://example.invalid")
    with state.owned():
        state.action("prepare")
        permit = state.admit_execution(command=CommandSpec(tmp_path, ("true",), timeout=5))
        state.observe_execution(permit, event="result", value=JobResult(0,
            facts=ExecutionFacts(disposition="unknown", cancellation_requested=True), job_references=(ref,)))
    calls = []
    def query(reference, **kwargs):
        calls.append(reference)
        return JobResult(0, facts=ExecutionFacts(disposition="terminal"))
    with state.owned():
        state.reconcile(SimpleNamespace(query_job=query))
        assert state.data["counts"]["dispatches"] == 1 and not state.pending()
    assert calls == [ref]


def test_physical_job_terminal_does_not_hide_unresolved_original_experiment(tmp_path):
    state = state_at(tmp_path)
    attempt = SimpleNamespace(attempt_id="original-attempt", disposition="unknown")
    record = SimpleNamespace(ref=ExperimentRef("cpu", "original"), lifecycle="running", attempts=(attempt,))
    campaign = SimpleNamespace(experiment_records=lambda: (record,))
    driver = SchemaWorkflow.__new__(SchemaWorkflow)
    driver.state, driver._original_pending = state, {}
    with state.owned():
        state.data["prepared"] = {"identity": "prepared-cpu"}
        driver._capture_unresolved(campaign)
        with pytest.raises(RecoveryRequired, match="Original Experiment"):
            driver._check_original_records(campaign, model=False)
        with pytest.raises(RecoveryRequired, match="Original Experiment"):
            driver._check_original_records(campaign, model=True)
        # Only the original request may finish a missing step once its old
        # attempt is truly reconciled; a fresh Coder still waits for settlement.
        attempt.disposition = "terminal"
        driver._check_original_records(campaign, model=False)
        with pytest.raises(RecoveryRequired, match="Original Experiment"):
            driver._check_original_records(campaign, model=True)
        record.lifecycle = "terminal"
        driver._check_original_records(campaign, model=True)
        assert state.data["unresolved_experiments"] == []
        assert state.data["counts"]["dispatches"] == state.data["counts"]["agent_calls"] == 0


def test_unknown_agent_requires_original_owner_and_missing_root_never_refunds(tmp_path):
    state = state_at(tmp_path, max_agent_calls=1)
    with state.owned():
        state.action("prepare")
        state.admit_agent(command=("fake",), workspace=tmp_path, timeout=5,
            environment={}, backend="claude", access=None, control=None)
    with state.owned():
        with pytest.raises(RecoveryRequired, match="unresolved"):
            state.reconcile(SimpleNamespace())
        state.reconcile(SimpleNamespace(), recover_agent=lambda *args: {"terminal": True, "basis": "fake owner confirmed stopped"})
        with pytest.raises(BudgetDenied):
            state.admit_agent(command=("fake",), workspace=tmp_path, timeout=5,
                environment={}, backend="claude", access=None, control=None)
        assert state.data["counts"]["agent_active_seconds"] == 5
    state.path.unlink()
    with pytest.raises(RecoveryRequired, match="root anchor"), state.owned():
        pass


def test_usage_uses_actual_call_identity_even_when_backend_reuses_session_id(tmp_path):
    state = state_at(tmp_path)
    driver = SchemaWorkflow.__new__(SchemaWorkflow)
    driver.state = state
    result = SimpleNamespace(session_id="provider-reused-id", terminal_usage=TokenUsage(4, 3, 0, 0, 7, "exact"))
    with state.owned():
        state.action("prepare")
        for _ in range(2):
            permit = state.admit_agent(command=("fake",), workspace=tmp_path, timeout=5,
                environment={}, backend="claude", access=None, control=None)
            state.observe_agent(permit, event="result", value={"active_seconds": .1, "result": ("", "", 0, False)})
            driver._native_usage(result, tmp_path)
        assert state.data["counts"]["tokens"] == 14
        assert len(state.data["usage_receipts"]) == 2
        for number in range(140):
            permit = state.admit_execution(command=CommandSpec(tmp_path, ("true",), timeout=1))
            state.observe_execution(permit, event="result", value=JobResult(0,
                facts=ExecutionFacts(disposition="terminal")))
        assert state.data["counts"]["dispatches"] == 140


def test_full_source_retention_preserves_deletions_and_detects_drift(tmp_path):
    state = state_at(tmp_path)
    owner = native(tmp_path).candidate_workspace
    with state.owned():
        original = RevisionContent({"kernel.py": b"# one", "removed.py": None, "new.cu": b"// two"})
        ref = state.retain_source(original, owner=owner, origin={"role": "draft"})
        assert state.read_source(ref, owner=owner).files == original.files
        path = state.directory / ref["object"]
        path.write_text(path.read_text().replace("IyBvbmU=", "IyB0d28="))
        with pytest.raises(RecoveryRequired, match="identity changed"):
            state.read_source(ref, owner=owner)


def test_legacy_real_run_bind_launcher_observations_and_replay_scope(tmp_path, monkeypatch):
    from aka.contracts.execution_environment import EnvironmentQuery
    from aka.execution.environment_query import _snapshot, match_environment
    fields = json.loads((Path(__file__).parent / "fixtures/agate_b41_environment.json").read_text())
    environment = _snapshot("legacy", "cpu-fixture", EnvironmentQuery("L20D"), fields["detail"],
        fields["capabilities"], [], LegacyExecutor.capabilities.operations)
    support = match_environment(EnvironmentQuery("L20D", "dev"), environment)
    monkeypatch.setattr(LegacyExecutor, "_environment_backend", lambda self: SimpleNamespace(check=lambda *a, **k: support))
    monkeypatch.setattr(LegacyExecutor, "_job_transport", lambda self: SimpleNamespace(environment={}, endpoint="", profile=""))
    events = []
    def run_owned(*args, **kw):
        events.append("launcher")
        result = JobResult(0, exit_code=0, facts=ExecutionFacts(disposition="terminal"))
        kw["on_observation"](result)
        return result
    monkeypatch.setattr("aka.execution.outcomes.run_owned", run_owned)
    executor = LegacyExecutor()
    command = CommandSpec(tmp_path, ("true",), hardware="L20D", operation="dev", timeout=5,
        on_observation=lambda result: events.append("native-owner"))
    with executor.dispatch_scope(admit=lambda **kw: events.append("admit") or "permit",
            observe=lambda permit, **kw: events.append(kw["event"])):
        bound = executor.bind(ExecutionContext({}, "L20D", 5))
        assert bound.run(command).exit_code == 0
        with pytest.raises(RuntimeError, match="replay"):
            bound.replay_job(command, idempotency_key="forbidden")
    assert events == ["admit", "launcher", "native-owner", "observation", "result"]
    with pytest.raises(RuntimeError, match="closed"):
        bound.run(command)


def test_limits_require_explicit_continuous_or_finite_budget():
    with pytest.raises(ValueError, match="finite mode"):
        RunLimits()
    limits = RunLimits(continuous=True)
    assert limits.max_windows is limits.max_episodes is limits.max_agent_calls is None
