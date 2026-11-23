"""CPU native owners -> frozen Schema inputs -> seal -> independent review."""
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aka.agent.provider import AgentOptions
from aka.contracts.execution import ExecutionContext as NativeContext
from aka.contracts.execution_environment import EnvironmentQuery, EnvironmentSnapshot
from aka.contracts.workspace import CandidateHandle, RevisionContent
from aka.execution.legacy.executor import LegacyExecutor
from aka.workflow.schema._native import RetainedJobEvidence, TaskConstraintReview, execution_context
from aka.workflow.schema.execution_context import CORRECTABLE, build_execution_context, sha256
from aka.workflow.schema.materials import read_design, snapshot_inputs, select_materials
from aka.workflow.schema.planner import Planner, PlannerConfig
from aka.workflow.schema.run_state import RunLimits, digest
from aka.workflow.schema.workflow import SchemaWorkflow
from test_schema_planner import CpuAgent, response
from test_schema_execution_context import authority, context, files, output, optimization
from test_schema_qualification_protocol import review
from test_schema_workflow import native


@pytest.fixture
def integrated(tmp_path, monkeypatch):
    monkeypatch.setattr("aka.execution.gateway.sandbox._find_agate", lambda: None)
    calls, actions, gets = [], [], []
    def get(url, method, path, payload, timeout, **kwargs):
        assert method == "GET" and payload is None
        gets.append(path)
        return {"gpu": "L20D", "arch": "sm_103", "source": "observed",
            "frameworks": {"triton": {"available": True, "version": "3.7.1"}}}
    monkeypatch.setattr("aka.execution.gateway.sandbox._gateway_json", get)
    # Isolate the actual backend cache between tests without querying anything.
    monkeypatch.setattr("aka.execution.environment_query._CACHE", {})
    def process(command, *, cwd, timeout, env):
        workspace = Path(cwd)
        calls.append(workspace)
        assert actions
        text = actions.pop(0)(workspace)
        return json.dumps({"type": "result", "result": text,
            "usage": {"input_tokens": 11, "output_tokens": 7}}), "", 0, False
    agent = CpuAgent(AgentOptions(state_directory=tmp_path / "agent"), process_runner=process,
        environment={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "ANTHROPIC_MODEL": "qwen3.8-max[1m]"})
    campaign = native(tmp_path)
    campaign.agent = agent
    campaign.handoff_resumes = 0
    executor = LegacyExecutor(configuration={"url": "https://cpu.invalid"})
    runtime = SimpleNamespace(executor=executor, execution_context=NativeContext({}, "L20D", 600))
    planner = Planner(agent, snapshot_inputs(tmp_path / "materials", select_materials(wiki_ids=[])),
        config=PlannerConfig(timeout_seconds=7, max_repairs=0))
    driver = SchemaWorkflow(tmp_path / "root", planner=planner, executor=executor,
        workspace=campaign.workspace, limits=RunLimits(continuous=True, retry_delays=()), sleep=lambda _: None)
    def enable():
        campaign.tool_runtime.workflow = runtime
        driver._bind_recovery(campaign)
    try:
        with driver.state.owned(), ExitStack() as stack:
            driver._original_pending = {}
            driver._campaign = campaign
            driver._scopes(stack, campaign)
            driver._prepare(campaign)
            yield SimpleNamespace(driver=driver, campaign=campaign, actions=actions,
                calls=calls, gets=gets, enable=enable, executor=executor)
    finally:
        agent.close()


def _coder(workspace):
    path = workspace / "kernel.py"
    path.write_bytes(path.read_bytes() + b"\n# new independently sealed D source\n")
    return "synthetic complete source"


def _seal(monkeypatch, campaign):
    from aka.workflow.light.long_horizon.campaign import LongHorizonCampaign
    from aka.workflow.light.long_horizon.models import EpisodeHandoff
    from aka.workflow.light.long_horizon.session import LongSessionRunner
    from aka.workflow.light.long_horizon.store import CampaignStore
    original, facts = LongSessionRunner.run, campaign.episode_facts
    sealed = {}
    def ready(self, workspace, *args, **kwargs):
        result = original(self, workspace, *args, **kwargs)
        _, active = CampaignStore.checkpoint(campaign.workspace)
        handle = CandidateHandle(active["episode"], active["base_commit"], active["episode_branch"], Path(active["worktree"]))
        source = campaign.candidate_workspace.read_source(workspace)
        commit = campaign.candidate_workspace.seal_candidate(handle, source)
        sealed[active["episode"]] = source
        return replace(result, exit_status=0, timed_out=False,
            handoff=EpisodeHandoff("candidate_ready", candidate_commit=commit), completion_diagnosis="")
    def source_facts(episode):
        value = facts(episode)
        value["candidate"] = sealed[episode]
        return value
    monkeypatch.setattr(LongSessionRunner, "run", ready)
    monkeypatch.setattr(campaign, "episode_facts", source_facts)
    monkeypatch.setattr(LongHorizonCampaign, "_completion_check", lambda *args: "")
    monkeypatch.setattr("aka.workflow.light.long_horizon.main_adapter.candidate_policy_violations", lambda *args, **kwargs: [])


def test_public_environment_c_e_d_and_review_share_original_bytes(integrated, monkeypatch):
    h = integrated
    h.enable()
    h.actions.append(lambda w: output(w, files(corrected=True)))
    h.driver._plan(h.campaign)
    state = h.driver.state.data
    frozen = deepcopy(state["planner_request"])
    ctx = frozen["execution_context_sha256"]
    assert len(h.gets) == 2  # Public native detail/capabilities GET, no job.
    assert state["counts"]["dispatches"] == 0
    publication = read_design(state_design(state))
    assert publication.manifest["version"] == 1
    assert json.loads(frozen["execution_context"])["architecture"]["source"] == "observed"
    # New E reuses the same prepared evidence, without discovery or a new hash.
    state["phase"], state["action"] = "optimize", None
    h.actions.append(lambda w: output(w, files(corrected=True, rewrite=True)))
    h.driver._plan(h.campaign)
    assert state["planner_request"]["execution_context_sha256"] == ctx and len(h.gets) == 2
    _seal(monkeypatch, h.campaign)
    h.actions.extend([_coder, lambda w: review(w, verdict="fail")])
    h.driver._implement(h.campaign)
    state = h.driver.state.data
    row = state["episodes"]["1"]
    assert row["binding"]["execution_context_sha256"] == ctx
    assert row["qualification_feedback"]["reviewed"]["execution_context_sha256"] == ctx
    assert row["qualification_feedback"]["execution_context_status"] == "bound"
    assert row["qualification_feedback"]["verdict"] == "fail" and not row["accepted"]
    assert row["sealed_candidate"]["identity"] != state["prepared"]["source_identity"]
    q = h.calls[-1] / "inputs"
    assert (q / "task/execution.json").read_text() == frozen["execution_context"]
    binding = json.loads((q / "review/binding.json").read_text())
    assert binding["source_identity"] == row["sealed_candidate"]["identity"]
    assert binding["source_identity"] != state["prepared"]["source_identity"]
    assert state["counts"]["agent_calls"] == 4 and state["counts"]["tokens"] == 72


def test_legacy_design_new_d_adopts_context_and_delivers_next_bound_review(integrated, monkeypatch):
    from aka.workflow.light.long_horizon.session import LongSessionRunner
    h = integrated
    h.actions.append(lambda w: output(w, files(corrected=True)))
    h.driver._plan(h.campaign)
    assert "execution_context" not in h.driver.state.data
    _seal(monkeypatch, h.campaign)
    prompts = []
    original_run = LongSessionRunner._run
    def run(self, workspace, prompt, **kwargs):
        prompts.append(prompt)
        return original_run(self, workspace, prompt, **kwargs)
    monkeypatch.setattr(LongSessionRunner, "_run", run)
    h.actions.extend([_coder, lambda w: review(w, verdict="fail")])
    h.driver._implement(h.campaign)
    old = deepcopy(h.driver.state.data)
    assert old["episodes"]["1"]["qualification_feedback"]["execution_context_status"] == "legacy"
    originals = {path: path.read_bytes() for path in (h.driver.directory / "operations").rglob("*") if path.is_file()}
    attempt = h.campaign.workspace / ".atrex_long_horizon/episodes/e0001/attempt.json"
    originals[attempt] = attempt.read_bytes()
    h.enable()
    engine = h.campaign.create_engine
    def interrupted(**kwargs):
        raise RuntimeError("synthetic interruption after binding, before admission")
    monkeypatch.setattr(h.campaign, "create_engine", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        h.driver._implement(h.campaign)
    # Recover solely from the durable state; a new D has adopted native facts,
    # while neither its admission nor any historical receipt has been rewritten.
    h.driver.state.data = json.loads(h.driver.state.path.read_bytes())
    state = h.driver.state.data
    binding = deepcopy(state["binding"])
    context = binding["execution_context_sha256"]
    assert context == state["execution_context"]["execution_context_sha256"]
    summary = json.loads(binding["context"].split("Unavailable feedback does not establish fidelity.\n", 1)[1])
    assert summary["reason"] == "qualification-binding-mismatch"
    assert state["counts"] == {**old["counts"], "episodes": 2}
    assert state["window"] == old["window"] and state["calls"] == old["calls"]
    gets = len(h.gets)
    assert gets == 2
    monkeypatch.setattr(h.driver, "_new_context", lambda *args: pytest.fail("adoption queried new facts"))
    monkeypatch.setattr(h.campaign, "create_engine", engine)
    reason = "The sealed source uses three stages while this design requires four."
    def failed_review(workspace):
        result = review(workspace, verdict="fail")
        path = workspace / "output/decision.json"
        decision = json.loads(path.read_text())
        for row in decision["obligations"]:
            row["reason"] = reason
        path.write_text(json.dumps(decision))
        return result
    h.actions.extend([_coder, failed_review])
    h.driver._implement(h.campaign)
    state = h.driver.state.data
    second = state["episodes"]["2"]
    assert second["binding"] == binding and binding["context"] in prompts[1]
    assert reason not in prompts[1]
    assert second["qualification_feedback"]["execution_context_status"] == "bound"
    assert second["qualification_feedback"]["reviewed"]["execution_context_sha256"] == context
    assert second["qualification_feedback"]["obligations"][0]["reason"] == reason
    assert not second["accepted"] and second["progress_assessment"]["reasons"] == ["qualification-failed"]
    selected = h.driver._qualification_feedback(h.campaign, seed_identity=state["working_draft"]["identity"])
    assert selected == second["qualification_feedback"]
    h.actions.extend([_coder, lambda w: review(w, verdict="fail")])
    h.driver._implement(h.campaign)
    state = h.driver.state.data
    third = state["episodes"]["3"]["binding"]
    delivered = json.loads(third["context"].split("Unavailable feedback does not establish fidelity.\n", 1)[1])
    assert delivered == selected and delivered["original_episode"] == 2
    assert third["context"] in prompts[2] and reason in prompts[2]
    assert third["execution_context_sha256"] == context and len(h.gets) == gets
    assert state["episodes"]["1"] == old["episodes"]["1"]
    assert all(path.read_bytes() == raw for path, raw in originals.items())
    for name in ("root_ref", "config", "prepared", "design", "design_version", "hypothesis", "hypotheses", "seen_records", "windows"):
        assert state[name] == old[name]
    assert all(state["calls"][key] == value for key, value in old["calls"].items())
    assert all(state["usage_receipts"][key] == value for key, value in old["usage_receipts"].items())
    assert state["window"]["episodes"] == [1, 2, 3] and state["window"]["progress"] == old["window"]["progress"]
    assert state["counts"]["agent_calls"] == 7 and state["counts"]["tokens"] == 126
    assert state["counts"]["dispatches"] == old["counts"]["dispatches"]
    (h.driver.directory / "audit.safe.json").write_text(json.dumps({
        "legacy_review_excluded_from_bound_d": True, "d2_reason_delivered_to_d3": True,
        "d2_d3_context_equal": True, "adoption_replay_without_discovery": True,
        "old_rows_inputs_and_usage_unchanged": True, "calls": 7, "tokens": 126,
        "native_discovery_gets": gets}) + "\n")


def state_design(state):
    from aka.workflow.schema.materials import DesignRef
    return DesignRef.from_dict(state["design"])


@pytest.mark.parametrize("rewrite", [False, True])
@pytest.mark.parametrize("device_label", ["L20D", "AKA L20D"])
def test_same_root_legacy_correction_retains_history_and_next_d(integrated, monkeypatch, rewrite, device_label):
    h = integrated
    def labeled_files(**kwargs):
        result = files(**kwargs)
        result["root.yaml"] = result["root.yaml"].replace(b"Device: L20D", ("Device: " + device_label).encode())
        return result
    h.actions.append(lambda w: output(w, labeled_files()))
    h.driver._plan(h.campaign)  # Original no-context publication.
    state = h.driver.state.data
    old = state_design(state)
    old_bytes = read_design(old).checked.files["root.yaml"]
    old_prepared, root = deepcopy(state["prepared"]), h.driver.state.root_ref
    draft = h.driver.state.retain_source(RevisionContent({"kernel.py": b"# existing working draft\n"}),
        owner=h.campaign.candidate_workspace, origin={"role": "working-draft"})
    state["working_draft"] = draft
    key = state["hypothesis"]
    state["hypotheses"][key].update(stagnant=2, uncertain_windows=2,
        seen_sources=["previously-seen"], milestones=["first-correct"], best_legal={"latency_us": 12})
    h.driver._open_window()
    window, history, counts = deepcopy(state["window"]), deepcopy(state["hypotheses"]), deepcopy(state["counts"])
    h.enable()
    h.driver.task_constraint_review = TaskConstraintReview(sha256(old_prepared["public_task"].encode()),
        {path: {"status": "unspecified", "value": None} for path in CORRECTABLE},
        **({"legacy_device_parent": old, "legacy_device_label": device_label,
            "legacy_device_target": "L20D"} if device_label != "L20D" else {}))
    state["phase"], state["action"] = "optimize", None
    def corrected(w):
        result = output(w, labeled_files(corrected=True, rewrite=rewrite))
        if rewrite:
            path = w / "output/decision.json"
            decision = json.loads(path.read_text())
            decision["hypothesis_key"] = "new-execution-rewrite"
            path.write_text(json.dumps(decision))
        return result
    h.actions.append(corrected)
    h.driver._plan(h.campaign)
    assert state["design_version"] == 2 and state_design(state).identity != old.identity
    assert read_design(state_design(state)).manifest["parent"] == old.as_dict()
    assert read_design(old).checked.files["root.yaml"] == old_bytes
    assert state["prepared"] == old_prepared and h.driver.state.root_ref == root
    assert state["counts"]["agent_calls"] == counts["agent_calls"] + 1
    assert state["counts"]["tokens"] == counts["tokens"] + 18
    assert state["window"] == window
    if rewrite:
        assert state["hypothesis"] == "new-execution-rewrite"
        assert state["hypotheses"][key] == history[key]
        assert state["working_draft"] == state["champion"]
    else:
        assert state["hypothesis"] == key and state["hypotheses"] == history
        assert state["working_draft"] == draft
    _seal(monkeypatch, h.campaign)
    h.actions.extend([_coder, lambda w: review(w, verdict="fail")])
    h.driver._implement(h.campaign)
    state = h.driver.state.data
    assert state["episodes"]["1"]["qualification_feedback"]["verdict"] == "fail"
    assert state["episodes"]["1"]["binding"]["design"]["identity"] == state["design"]["identity"]
    assert state["episodes"]["1"]["sealed_candidate"]["identity"] != old_prepared["source_identity"]
    binding = state["episodes"]["1"]["binding"]
    assert binding["execution_context_sha256"] == state["execution_context"]["execution_context_sha256"]
    sealed = state["episodes"]["1"]["sealed_candidate"]["identity"]
    assert h.driver._qualification_feedback(h.campaign, seed_identity=sealed,
        context_binding=binding)["status"] == "reviewed"
    assert h.driver._qualification_feedback(h.campaign, seed_identity="different-new-source",
        context_binding=binding)["status"] == "unavailable"
    assert h.driver._qualification_feedback(h.campaign, seed_identity=sealed,
        context_binding={"execution_context_sha256": "0" * 64})["status"] == "unavailable"
    assert read_design(state_design(state)).checked.resolved.raw_task["root"]["CONTRACT"]["Hardware_specification"]["Device"] == device_label
    # The attestation survives the next ordinary E keep without creating a
    # version, a hypothesis or a replacement context for D/reviewer.
    retained_context, retained_design = deepcopy(state["execution_context"]), deepcopy(state["design"])
    # The earlier synthetic history deliberately exhausted this hypothesis.
    # Exercise ordinary keep with an eligible hypothesis after proving that
    # correction itself retained the exhausted counters unchanged above.
    state["hypotheses"][state["hypothesis"]].update(stagnant=0, uncertain_windows=0)
    state["phase"], state["action"] = "optimize", None
    h.actions.append(lambda w: response(w, status="keep"))
    h.driver._plan(h.campaign)
    assert state["phase"] == "implement" and state["design"] == retained_design
    assert state["execution_context"] == retained_context


def test_frozen_request_replay_uses_original_context_without_discovery(integrated, monkeypatch):
    h = integrated
    h.enable()
    h.actions.append(lambda w: output(w, files(corrected=True)))
    original = h.driver.planner.design
    def lost(request):
        result = original(request)
        raise RuntimeError("synthetic lost return")
    monkeypatch.setattr(h.driver.planner, "design", lost)
    with pytest.raises(RuntimeError, match="lost return"):
        h.driver._plan(h.campaign)
    frozen = deepcopy(h.driver.state.data["planner_request"])
    monkeypatch.setattr(h.driver.planner, "design", original)
    monkeypatch.setattr(h.driver, "_new_context", lambda *args: pytest.fail("replay queried new facts"))
    h.driver._plan(h.campaign)
    assert h.driver.state.data["planner_request"] == frozen
    assert len(h.calls) == 1 and len(h.gets) == 2


def test_legacy_device_keep_adopts_reviewed_label_without_a_correction(integrated, monkeypatch):
    from test_schema_execution_context import labeled_files
    h = integrated
    h.actions.append(lambda w: output(w, labeled_files(corrected=True)))
    h.driver._plan(h.campaign)
    state = h.driver.state.data
    parent, before = state_design(state), deepcopy(state["counts"])
    h.enable()
    h.driver.task_constraint_review = TaskConstraintReview(sha256(state["prepared"]["public_task"].encode()),
        {path: {"status": "unspecified", "value": None} for path in CORRECTABLE}, parent, "AKA L20D", "L20D")
    state["phase"], state["action"] = "optimize", None
    h.actions.append(lambda w: response(w, status="keep"))
    h.driver._plan(h.campaign)
    assert state["phase"] == "implement" and state_design(state) == parent
    assert "contract_correction_sha256" not in state["planner_request"]
    assert state["counts"]["agent_calls"] == before["agent_calls"] + 1
    context_hash = state["execution_context"]["execution_context_sha256"]
    _seal(monkeypatch, h.campaign)
    h.actions.extend([_coder, lambda w: review(w, verdict="fail")])
    h.driver._implement(h.campaign)
    row = h.driver.state.data["episodes"]["1"]
    assert row["binding"]["execution_context_sha256"] == context_hash
    assert row["qualification_feedback"]["verdict"] == "fail"


@pytest.mark.parametrize("source", ["observed", "configured", "declared", "cache", "unknown"])
def test_public_snapshot_provenance_is_not_inferred_from_campaign_arch(integrated, monkeypatch, source):
    h = integrated
    h.enable()
    h.campaign.arch = "sm_89"  # Native prompt/config can override; it is not observed evidence.
    route = h.driver._recovery_executor._environment_backend()
    observed = {"gpu": "L20D", "arch": "sm_103", "toolchain": {"triton": "3.7.1"}}
    snapshot = EnvironmentSnapshot(route.backend, route.endpoint, "L20D", None,
        declarations={"environment": observed} if source == "declared" else {},
        observed={} if source == "declared" else observed, source=source)
    monkeypatch.setattr(h.driver._recovery_executor, "query_environment", lambda q: snapshot)
    ctx = h.driver._new_context(h.campaign, "provenance")
    assert ctx.as_dict()["architecture"]["source"] == source
    assert ctx.as_dict()["architecture"]["value"] == (None if source == "unknown" else "sm_103")
    assert all(value["status"] == "unknown" for value in ctx.as_dict()["task_constraints"].values())


@pytest.mark.parametrize("name", ["Triton", "CuteDSL", "Cuda", "TileLang", "FlyDSL", "cute"])
def test_native_framework_selection_preserves_original_policy_hash(name):
    from aka.task.source.inspection import framework_key
    policy = {"framework": name, "sandbox_hardware": "L20D"}
    ctx = build_execution_context(prepared_id="fixture", public_task="public", target_hardware="L20D",
        native_policy=policy)
    assert ctx.as_dict()["selected_framework"]["value"] == framework_key(name)
    assert ctx.as_dict()["binding"]["native_policy_sha256"] == digest(policy)


@pytest.mark.parametrize("change", ["drop", "policy", "task", "target"])
def test_plain_optimize_cannot_drop_or_rebind_parent_authority(authority, change):
    h = authority
    ctx = context(h)
    h.actions.append(lambda w, e: output(w, files(corrected=True)))
    parent = h.planner.design(replace(h.request, execution_context=ctx)).design_ref
    if change == "policy":
        ctx = context(h, policy={"framework": "triton", "sandbox_hardware": "L20D", "verify_repeats": 9})
    elif change == "task":
        h.request = replace(h.request, public_task="Changed external Task")
        ctx = context(h)
    elif change == "target":
        h.request = replace(h.request, target_hardware="Different GPU")
        ctx = context(h)
    elif change == "drop":
        ctx = None
    with pytest.raises(ValueError, match="parent"):
        h.planner.optimize(optimization(h, parent, ctx))
    assert len(h.calls) == 1


def test_plain_optimize_accepts_new_context_with_same_external_binding(authority):
    h = authority
    initial = context(h)
    h.actions.append(lambda w, e: output(w, files(corrected=True)))
    parent = h.planner.design(replace(h.request, execution_context=initial)).design_ref
    refreshed = context(h, task_status="unknown")
    assert refreshed.sha256 != initial.sha256
    assert refreshed.as_dict()["binding"] == initial.as_dict()["binding"]
    h.actions.append(lambda w, e: output(w, files(corrected=True, rewrite=True)))
    result = h.planner.optimize(optimization(h, parent, refreshed))
    assert result.status == "replace" and result.contract_correction is None
    assert json.loads((result.directory / "intent.json").read_text())["inputs"]["task/execution.json"] == refreshed.sha256


@pytest.mark.parametrize("contradiction", [False, True])
def test_keep_retains_published_context_unless_new_facts_conflict(integrated, monkeypatch, contradiction):
    h = integrated
    h.enable()
    state = h.driver.state.data
    task_review = TaskConstraintReview(sha256(state["prepared"]["public_task"].encode()),
        {path: {"status": "unspecified", "value": None} for path in CORRECTABLE})
    h.driver.task_constraint_review = task_review
    h.actions.append(lambda w: output(w, files(corrected=True)))
    h.driver._plan(h.campaign)
    parent = state_design(state)
    parent_context = state["execution_context"]["execution_context_sha256"]
    route = h.driver._recovery_executor._environment_backend()
    snapshot = EnvironmentSnapshot(route.backend, route.endpoint, "L20D", None,
        observed={"gpu": "L20D", "arch": "sm_104" if contradiction else "sm_103",
                  "toolchain": {"triton": "3.7.1"}}, source="observed", probed_at="new CPU observation")
    monkeypatch.setattr(h.driver._recovery_executor, "query_environment", lambda q: snapshot)
    refreshed = execution_context(campaign=h.campaign, prepared=state["prepared"], executor=h.driver._recovery_executor,
        directory=h.driver.directory / "new-observation", task_review=task_review)
    assert refreshed.sha256 != parent_context
    monkeypatch.setattr(h.driver, "_new_context", lambda *args: refreshed)
    state["phase"], state["action"] = "optimize", None
    h.actions.append(lambda w: response(w, status="keep"))
    h.driver._plan(h.campaign)
    assert state["planner_request"]["execution_context_sha256"] == refreshed.sha256
    assert state["design_version"] == 1 and state_design(state) == parent
    assert state["execution_context"]["execution_context_sha256"] == parent_context
    if contradiction:
        assert state["phase"] == "optimize" and state["planner_failures"] == 1
        corrected = files(corrected=True)
        corrected["root.yaml"] = corrected["root.yaml"].replace(b"Architecture: sm_103", b"Architecture: sm_104")
        h.actions.append(lambda w: output(w, corrected))
        h.driver._plan(h.campaign)
        assert state["design_version"] == 2 and state["phase"] == "implement"
        implementation_context = refreshed.sha256
    else:
        assert state["phase"] == "implement" and state["planner_failures"] == 0
        implementation_context = parent_context
    _seal(monkeypatch, h.campaign)
    h.actions.extend([_coder, lambda w: review(w, verdict="fail")])
    h.driver._implement(h.campaign)
    state = h.driver.state.data
    assert state["episodes"]["1"]["binding"]["execution_context_sha256"] == implementation_context
    assert state["episodes"]["1"]["qualification_feedback"]["reviewed"]["execution_context_sha256"] == implementation_context
    assert state["episodes"]["1"]["qualification_feedback"]["verdict"] == "fail"
    assert state["counts"]["agent_calls"] == (5 if contradiction else 4)
    assert state["counts"]["tokens"] == (90 if contradiction else 72)


@pytest.fixture
def retained_native(tmp_path):
    from aka.contracts.execution import JobResult, ExecutionFacts
    from aka.contracts.execution_recovery import JobReference
    from aka.evaluation.gateway_command import GatewayCommandEvaluator
    from aka.execution.gateway.requests import operation_request
    from aka.execution.live_eval import WIRE_PATH, digest as wire_digest
    from aka.experiment.file_records import FileExperimentRecords
    from aka.experiment.runner import LegacyExperiment
    from test_native_environment_compat import request, raw_evaluation
    command, task = request(tmp_path / "native")
    command = replace(command, environment={"AGATE_AK": "synthetic-namespace"}, idempotency_key="synthetic-request")
    executor = LegacyExecutor(configuration={"url": "https://cpu.invalid"}).bind(
        NativeContext({"AGATE_AK": "synthetic-namespace"}, "L20D", 600))
    records = FileExperimentRecords(tmp_path / "records", str(command.workspace.absolute()))
    bodies = []
    def execute(command):
        record, = records.iter_records()
        attempt = record.attempts[-1]
        prepared = json.loads(command.submission)
        wire = operation_request(prepared["request"], "run", ssh=False)
        wire["idempotency_key"] = command.idempotency_key
        wire.setdefault("options", {})["timeout_s"] = command.timeout
        ref = JobReference("gateway", "cpu-owned-job", endpoint="https://cpu.invalid",
            request_identity=wire_digest(wire), idempotency_key=command.idempotency_key)
        records.publish_content(record.ref, json.dumps({"attempt_id": attempt.attempt_id,
            "terminal": True, "job_reference": {"identifier": ref.identifier},
            "request_identity": ref.request_identity,
            "route_identity": wire_digest([ref.endpoint, WIRE_PATH, "", ""]),
            "namespace_identity": wire_digest("synthetic-namespace")}).encode(), "live_eval_association")
        original = json.loads(records.read_artifact(next(a for a in record.artifacts if a.role == "evaluation_input")))
        hashes = {name: sha256(value.encode()) for name, value in original["reference_files"].items()
                  if name in {"input.py", "reference.py", "metadata.json", "shapes.json"}}
        hashes["candidate.py"] = sha256(wire["candidate"].encode())
        body = {"job_id": ref.identifier, "status": "succeeded", "num_gpus": 1,
            "request_digest": {"gpu": "NVIDIA L20D", "num_gpus": 1},
            "result": {**raw_evaluation(), "dsl": "triton",
                "environment": {"gpu_name": "NVIDIA L20D", "gpu_arch": "sm_103", "triton_version": "3.7.1"},
                "shape_selection": {"all_shape_ids": list(task.case_ids), "partial": False, "source_hashes": hashes}}}
        raw = json.dumps(body).encode()
        bodies.append(raw)
        snapshot = EnvironmentSnapshot("legacy", ref.endpoint, "L20D", None,
            observed={"gpu": "L20D", "arch": "sm_103", "toolchain": {"triton": "3.7.1"}}, source="cache")
        return JobResult(0, stdout=raw.decode(), transport_status="succeeded",
            facts=ExecutionFacts(disposition="terminal"), environment_snapshot=snapshot, job_references=(ref,))
    evaluator = GatewayCommandEvaluator(execution_capabilities=LegacyExecutor.capabilities, evaluation_input=task)
    experiment = LegacyExperiment(SimpleNamespace(capabilities=LegacyExecutor.capabilities, run=execute),
        records=records, command_evaluator=evaluator)
    record = experiment.run_recorded(command, request_id="fixture-native")
    prepared = {"identity": "prepared-cpu", "public_task": "Complete synthetic public Task", "hardware": "L20D",
        "native_policy": {"framework": "Triton", "sandbox_hardware": "L20D"}}
    campaign = SimpleNamespace(workspace=command.workspace, problem=SimpleNamespace(evaluation=task),
        problem_provider=SimpleNamespace(snapshot=lambda p: SimpleNamespace(identity="prepared-cpu")),
        experiment_record_reader=lambda: nullcontext(records))
    evidence = RetainedJobEvidence(bodies[0], sha256(bodies[0]), record.ref, digest(record.as_dict()))
    return SimpleNamespace(campaign=campaign, prepared=prepared, executor=executor, records=records,
        record=record, evidence=evidence, directory=tmp_path / "operation")


def test_original_native_owner_and_source_evidence_projects_without_query(retained_native, monkeypatch):
    h = retained_native
    monkeypatch.setattr(h.executor, "query_environment", lambda q: pytest.fail("retained evidence queried live"))
    review = TaskConstraintReview(sha256(h.prepared["public_task"].encode()),
        {path: {"status": "unspecified", "value": None} for path in CORRECTABLE})
    ctx = execution_context(campaign=h.campaign, prepared=h.prepared, executor=h.executor,
        directory=h.directory, retained_job=h.evidence, task_review=review)
    assert ctx.as_dict()["architecture"] == {"value": "sm_103", "source": "observed",
        "evidence_sha256": h.evidence.raw_sha256}
    assert ctx.as_dict()["toolchain"]["value"] == "3.7.1"
    assert b"cpu.invalid" not in ctx.as_bytes() and b"synthetic-namespace" not in ctx.as_bytes()
    # Ordinary retained preparation snapshots keep their original cache source.
    cached = execution_context(campaign=h.campaign, prepared=h.prepared, executor=h.executor,
        directory=h.directory, preparation_records=(h.record.ref,))
    assert cached.as_dict()["architecture"]["source"] == "cache"


@pytest.mark.parametrize("framework,expected", [("Cuda", "13.0.88"), ("pytorch", "2.9.0+cu130")])
def test_native_provider_toolchain_format_builds_context(retained_native, monkeypatch, framework, expected):
    import importlib.metadata
    import importlib.util
    import shutil
    import subprocess
    import sys
    from aka.execution.ssh.environment import environment
    from aka.execution.environment_query import _snapshot
    h = retained_native
    torch = SimpleNamespace(cuda=SimpleNamespace(
        get_device_properties=lambda _: SimpleNamespace(name="NVIDIA L20D", multi_processor_count=128, total_memory=48 * 2**30),
        get_device_capability=lambda _: (10, 3)), version=SimpleNamespace(hip=None, cuda="13.0"), __version__="2.9.0+cu130")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: ())
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "1.0.0")
    monkeypatch.setattr(shutil, "which", lambda name: "/cpu/nvcc" if name == "nvcc" else None)
    banner = ("nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2025 NVIDIA Corporation\n"
              "Cuda compilation tools, release 13.0, V13.0.88\nBuild cuda_13.0.r13.0/compiler.36424714_0")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=banner, returncode=0))
    observed = environment({"hardware": "L20D"})
    assert observed["toolchain"]["nvcc"]["version"] == banner
    route = h.executor._environment_backend()
    snapshot = _snapshot(route.backend, route.endpoint, EnvironmentQuery("L20D"), observed, observed, [], route.operations)
    attempt = h.record.attempts[-1]
    revised = replace(h.record, attempts=(*h.record.attempts[:-1],
        replace(attempt, result=replace(attempt.result, environment_snapshot=snapshot))))
    original_get = h.records.get
    monkeypatch.setattr(h.records, "get", lambda ref: revised if ref == h.record.ref else original_get(ref))
    monkeypatch.setattr(h.executor, "query_environment", lambda q: pytest.fail("native snapshot queried again"))
    prepared = deepcopy(h.prepared)
    prepared["native_policy"]["framework"] = framework
    ctx = execution_context(campaign=h.campaign, prepared=prepared, executor=h.executor, directory=h.directory,
        preparation_records=(h.record.ref,))
    doc = ctx.as_dict()
    assert doc["toolchain"]["value"] == expected and doc["toolchain"]["source"] == "observed"
    assert doc["architecture"]["value"] == "sm_103"
    assert doc["binding"]["native_policy_sha256"] == digest(prepared["native_policy"])
    assert all(value["status"] == "unknown" for value in doc["task_constraints"].values())
    assert b"nvcc: NVIDIA" not in ctx.as_bytes() and b"/cpu" not in ctx.as_bytes()
    assert original_get(h.record.ref) == h.record
    (h.directory.parent / "nvcc-audit.safe.json").write_text(json.dumps({
        "native_provider_formatter_used": True, "observed_native_toolchain_accepted": True,
        "private_banner_excluded": True, "original_record_unchanged": True,
        "real_gpu_calls": 0, "real_subprocess_calls": 0, "model_calls": 0}) + "\n")


@pytest.mark.parametrize("framework,value,expected", [
    ("cuda", "Cuda compilation tools, release 12.8, V12.8.93", "12.8.93"),
    ("cuda", "Cuda compilation tools, release 12.8, V13.0.1", None),
    ("cuda", "Cuda compilation tools, release 12.8, V12.8.93\nCuda compilation tools, release 13.0, V13.0.88", None),
    ("cuda", "unrecognized banner 13.0.88", None),
    ("cuda", "/private/toolchain-13.0.88", None),
    ("cuda", "https://private.invalid/version/13.0.88", None),
    ("triton", "3.7.1", "3.7.1"),
    ("triton", "3.7.1 PRIVATE_SENTINEL", None),
    ("pytorch", "2.9.0+cu130", "2.9.0+cu130"),
    ("pytorch", "2.9.0+cpu", "2.9.0+cpu"),
    ("pytorch", "2.9.0+private", None),
])
def test_native_version_projection_is_bounded_and_preserves_provenance(framework, value, expected):
    from aka.workflow.schema._native import _version
    name = {"cuda": "nvcc", "pytorch": "torch"}.get(framework, framework)
    for source in ("observed", "configured", "declared", "cache", "unknown"):
        token, provenance = _version({name: value}, framework, source)
        assert token == (expected if source != "unknown" else None)
        assert provenance == (source if token is not None else "unknown")
    assert _version({name: {"version": value, "source": "declared"}}, framework, "observed") == (
        expected, "declared" if expected is not None else "unknown")


@pytest.mark.parametrize("damage", ["raw-hash", "record-hash", "prepared", "task", "hardware", "route", "namespace", "policy", "source", "shape"])
def test_wrong_native_binding_is_rejected(retained_native, damage):
    h = retained_native
    evidence, prepared, executor = h.evidence, deepcopy(h.prepared), h.executor
    if damage == "raw-hash":
        evidence = replace(evidence, raw_sha256="0" * 64)
    elif damage == "record-hash":
        evidence = replace(evidence, record_sha256="0" * 64)
    elif damage == "prepared":
        prepared["identity"] = "different-prepared"
    elif damage == "task":
        h.campaign.problem.evaluation = replace(h.campaign.problem.evaluation, case_ids=("another-shape",))
    elif damage == "hardware":
        prepared["hardware"] = prepared["native_policy"]["sandbox_hardware"] = "another GPU"
    elif damage in {"route", "namespace"}:
        executor = LegacyExecutor(configuration={"url": "https://other.invalid" if damage == "route" else "https://cpu.invalid"}).bind(
            NativeContext({"AGATE_AK": "other-namespace" if damage == "namespace" else "synthetic-namespace"}, "L20D", 600))
    elif damage == "policy":
        prepared["native_policy"]["framework"] = "Cuda"
    else:
        body = json.loads(evidence.raw)
        if damage == "source":
            body["result"]["shape_selection"]["source_hashes"]["candidate.py"] = "0" * 64
        else:
            body["result"]["shape_selection"]["all_shape_ids"] = ["other-shape"]
        raw = json.dumps(body).encode()
        evidence = replace(evidence, raw=raw, raw_sha256=sha256(raw))
    with pytest.raises(ValueError):
        execution_context(campaign=h.campaign, prepared=prepared, executor=executor,
            directory=h.directory, retained_job=evidence)
