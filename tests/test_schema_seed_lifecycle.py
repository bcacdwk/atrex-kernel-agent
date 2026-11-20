"""A4 CPU only: real Task bytes/native Store/Git/drafts, synthetic fees/timings.

This fixture never calls A2's CPU result a native/GPU PASS. Its local test
Evaluation executes the real FP32 softmax+copy V0 but supplies synthetic native
timings solely to exercise the native lifecycle beyond the V0 boundary.
"""
import json
from pathlib import Path
from uuid import UUID

import pytest
import torch

from aka.task.source.legacy import LegacySourceProvider
from aka.workflow.light.long_horizon.store import CampaignStore
from aka.workflow.light.orchestration.constants import TEST_RESULT_PREFIX
from aka.workflow.schema.authority import NativeCallAuthority
from aka.workflow.schema.context import RoleContext
from aka.workflow.schema.preparation import NAMESPACE, SeedLifecycle
from aka.workflow.schema.session import native_agent, RoleSession
from aka.workflow.schema.task import SchemaProblemProvider
from kernelschema.api import open_run, prepare_run
from kernelschema.artifacts import Publication
from kernelschema.budget import OwnerAccount, cost
from kernelschema.contracts import ProtocolError, UnsupportedError, jcs, sha256

from schema_cpu_fixture import Owner, make_engine
from test_schema_optional_module import fixed, profile, fake_runner


def engine_for(fixed, tmp_path, *, reviewers=False):
    campaign, engine = make_engine(tmp_path)
    q = fixed.run.request
    campaign.problem_provider = SchemaProblemProvider(q["task"], q["scope"], q["materials"])
    campaign.problem = campaign.problem_provider.load(fixed.run.root,
        snapshot_directory=tmp_path / "native-problem")
    campaign.source_provider = LegacySourceProvider()
    campaign.v1_ask_codex = reviewers
    def synthetic_evaluation(workspace, argv, **kwargs):
        source = (workspace / "kernel.py").read_bytes()
        campaign.evaluations.append((tuple(argv), source))
        if campaign.evaluation_failure:
            raise campaign.evaluation_failure
        namespace = {}
        exec(compile(source, "real_task_baseline", "exec"), namespace)
        x = torch.tensor([[1, 2, 3], [2, 2, 2]], dtype=torch.float32)
        out = torch.full_like(x, float("nan"))
        assert namespace["Model"]()(x, out) is out
        torch.testing.assert_close(out, torch.softmax(x, dim=-1))
        result = {"all_pass": True, "latency_us_geomean": 10.0,
                  "latency_us_by_shape": {"synthetic": 10.0}}
        return {"exit_code": 0, "stdout": TEST_RESULT_PREFIX + json.dumps(result), "stderr": ""}
    campaign.measure_for_acceptance = synthetic_evaluation
    owner = Owner(tmp_path / "owner")
    lifecycle = SeedLifecycle.create(fixed.run, engine, owner)
    return campaign, engine, owner, lifecycle


def change_seed(fixed, tmp_path, data):
    p = Publication(fixed.run.root)
    seed = p.put_source("explicit-human-partial", data)
    request = dict(fixed.run.request, seed=seed)
    q = p.put_json("seed-request.json", request)
    p.commit([q])
    (fixed.run.root / "seed-request.json").write_bytes(jcs(request))
    fixed.run = prepare_run("implement", request=fixed.run.root / "seed-request.json",
        out=tmp_path / "explicit-run", backend_identity={"synthetic": "A4 CPU only"})


@pytest.mark.parametrize("explicit", [False, True])
def test_actual_v0_then_exact_c0_and_unpromoted_c1_to_e2(fixed, tmp_path, explicit):
    if explicit:
        change_seed(fixed, tmp_path, b"# human partial \xce\xb1\r\ndef forward(x, out):\r\n    pass")
    campaign, engine, owner, lifecycle = engine_for(fixed, tmp_path)
    c0 = fixed.run.bundle.source(lifecycle.current_source().source)
    first = lifecycle.run_episode(1)
    assert first.attempt["status"] == "invalid_handoff"
    assert first.attempt["accepted"] is False
    assert campaign.evaluations[0][1] == fixed.run.bundle.source(fixed.task["baseline"])
    assert campaign.agent.calls[0]["received"] == c0
    c1 = c0 + b"\n# synthetic coder progress\n"
    assert fixed.run.bundle.source(lifecycle.current_source().source) == c1
    state = fixed.run.load_state(NAMESPACE)
    e1 = state["episodes"]["1"]
    assert state["v0"]["revision"] != e1["intent"]["base_commit"]
    assert e1["incumbent"]["source"]["sha256"] == state["binding"]["baseline"]["source"]["sha256"]
    before = campaign.candidate_workspace.current_revision(campaign.workspace)
    assert before != e1["intent"]["base_commit"]
    assert campaign.candidate_workspace.read_source(campaign.workspace).files["kernel.py"] == campaign.evaluations[0][1]
    reopened = SeedLifecycle.reopen(open_run("implement", fixed.run.root), engine, owner)
    second = reopened.run_episode(2)
    assert second.attempt["accepted"] is False
    assert campaign.agent.calls[1]["received"] == c1
    state = fixed.run.load_state(NAMESPACE)
    assert state["episodes"]["2"]["intent"]["base_commit"] == before
    assert state["episodes"]["2"]["seed"]["source"]["sha256"] == state["episodes"]["1"]["current"]["source"]["sha256"]
    assert len(campaign.evaluations) == campaign.setup_count == 1
    reopened.run_episode(1)  # Replay older terminal does not rewind latest C.
    assert len(campaign.agent.calls) == 2 and fixed.run.load_state(NAMESPACE)["latest"]["episode"] == 2


@pytest.mark.parametrize("boundary", ["intent", "during_install", "ready", "admitted"])
def test_seed_crashes_resume_original_transaction(fixed, tmp_path, monkeypatch, boundary):
    campaign, engine, owner, lifecycle = engine_for(fixed, tmp_path)
    original_save = lifecycle._save
    armed = True
    def save(state):
        nonlocal armed
        original_save(state)
        if armed and state["episodes"].get("1", {}).get("phase") == boundary:
            armed = False
            raise KeyboardInterrupt("synthetic power loss after " + boundary)
    monkeypatch.setattr(lifecycle, "_save", save)
    install = campaign.candidate_workspace.install_source
    def partial(workspace, source):
        nonlocal armed
        if boundary == "during_install" and armed and "worktrees" in str(workspace):
            armed = False
            (workspace / "kernel.py").write_bytes(b"# incomplete host write")
            raise KeyboardInterrupt("synthetic power loss during installation")
        return install(workspace, source)
    monkeypatch.setattr(campaign.candidate_workspace, "install_source", partial)
    with pytest.raises(KeyboardInterrupt):
        lifecycle.run_episode(1)
    state = fixed.run.load_state(NAMESPACE)
    assert not campaign.agent.calls and len(campaign.evaluations) == 1
    assert CampaignStore.checkpoint(campaign.workspace)[1]["phase"] == "preparing"
    expected = fixed.run.bundle.source(state["binding"]["seed"]["source"])
    if boundary == "admitted":
        draft = Path(state["episodes"]["1"]["draft"])
        expected += b"\n# existing user progress after admission\n"
        (draft / "kernel.py").write_bytes(expected)
    owner.denied.add("episode")
    reopened = SeedLifecycle.reopen(fixed.run, engine, owner)
    assert reopened.run_episode(1).reason == "external-admission: episode"
    owner.denied.clear()
    reopened.run_episode(1)
    assert campaign.agent.calls[0]["received"] == expected
    assert len(campaign.evaluations) == 1 and len(campaign.agent.calls) == 1


def pause_after_admission(fixed, tmp_path, monkeypatch):
    campaign, engine, owner, lifecycle = engine_for(fixed, tmp_path)
    original = lifecycle.draft_ready
    def stop(**kwargs):
        original(**kwargs)
        raise KeyboardInterrupt("before initial coder")
    monkeypatch.setattr(lifecycle, "draft_ready", stop)
    with pytest.raises(KeyboardInterrupt):
        lifecycle.run_episode(1)
    return campaign, engine, owner, SeedLifecycle.reopen(fixed.run, engine, owner)


def test_exploring_resume_preserves_draft_and_refuses_controller_drift(fixed, tmp_path, monkeypatch):
    campaign, engine, owner, lifecycle = engine_for(fixed, tmp_path)
    campaign.agent.fail = KeyboardInterrupt("synthetic interrupted native coder")
    with pytest.raises(KeyboardInterrupt):
        lifecycle.run_episode(1)
    state = fixed.run.load_state(NAMESPACE)
    draft = Path(state["episodes"]["1"]["draft"])
    progress = (draft / "kernel.py").read_bytes() + b"\n# user progress\n"
    (draft / "kernel.py").write_bytes(progress)
    owner.denied.add("recover")
    assert lifecycle.run_episode(1).reason == "external-admission: recover"
    owner.denied.clear()
    lifecycle.run_episode(1)
    assert campaign.agent.calls[1]["received"] == progress
    assert len(campaign.evaluations) == 1


@pytest.mark.parametrize("damage", ["delete", "extra", "metadata", "mode", "state", "controller"])
def test_admitted_corruption_fails_without_overwriting_draft(fixed, tmp_path, monkeypatch, damage):
    campaign, engine, owner, lifecycle = pause_after_admission(fixed, tmp_path, monkeypatch)
    state = fixed.run.load_state(NAMESPACE)
    entry = state["episodes"]["1"]
    draft = Path(entry["draft"])
    if damage == "delete":
        (draft / "kernel.py").unlink()
    elif damage == "extra":
        (draft / "extra.py").write_text("old auxiliary source")
    elif damage == "metadata":
        (draft / "solution.json").write_text('{"sources": ["kernel.py", "extra.py"]}')
    elif damage == "mode":
        (draft / "kernel.py").chmod(0o755)
    elif damage == "state":
        (draft.parent / "state.json").unlink()
    else:
        (Path(entry["intent"]["worktree"]) / "kernel.py").write_bytes(b"controller changed")
    before = (draft / "kernel.py").read_bytes() if (draft / "kernel.py").exists() else None
    with pytest.raises((ProtocolError, FileNotFoundError)):
        lifecycle.run_episode(1)
    assert not campaign.agent.calls
    assert ((draft / "kernel.py").read_bytes() if (draft / "kernel.py").exists() else None) == before
    assert fixed.run.load_state(NAMESPACE)["episodes"]["1"]["current"] == entry["current"]


def test_lost_seed_state_and_changed_request_do_not_guess(fixed, tmp_path):
    campaign, engine, owner, lifecycle = engine_for(fixed, tmp_path)
    lifecycle.run_episode(1)
    path = fixed.run.root / f".runtime/state/{NAMESPACE}.json"
    original = path.read_bytes()
    path.unlink()
    with pytest.raises(ProtocolError, match="cannot read"):
        SeedLifecycle.reopen(fixed.run, engine, owner)
    with pytest.raises(ProtocolError, match="existing native work"):
        SeedLifecycle.create(fixed.run, engine, owner)
    path.write_bytes(original)
    engine.token_budget += 1
    with pytest.raises(ProtocolError, match="binding changed"):
        SeedLifecycle.reopen(fixed.run, engine, owner)
    assert len(campaign.evaluations) == 1


@pytest.mark.parametrize("fault", [None, "publishing", "installing", "after_install", "partial"])
def test_actual_a3_result_installation_once_and_full_replacement(fixed, tmp_path, monkeypatch, fault):
    campaign, engine, owner, lifecycle = pause_after_admission(fixed, tmp_path, monkeypatch)
    p, trace = profile(tmp_path), []
    ledger = OwnerAccount.for_context(fixed.run, create=True)
    agent = native_agent(p, process_runner=fake_runner(trace))
    source = lifecycle.current_source(1).source
    reservation = ledger.reserve(fixed.run.run_id, "a4-coder", role="coder", purpose="A4 synthetic native call",
        amount=cost(role="coder", calls=1, active_ms=8000))
    authority = NativeCallAuthority.bind(fixed.run, ledger, reservation, source=source, episode=1)
    role = RoleContext.prepare(fixed.run, "coder", source=source)
    scope = tmp_path / "role-scope"
    scope.mkdir()
    session = RoleSession.from_authority(agent, role, p, authority=authority, workspace=scope,
        session_id=str(UUID(int=41)), output_directory=tmp_path / "role-output")
    try:
        result = session.invoke()
        state = fixed.run.load_state(NAMESPACE)
        draft = Path(state["episodes"]["1"]["draft"])
        last_complete = lifecycle.current_source(1)
        armed = True
        original_save = lifecycle._save
        original_install = campaign.candidate_workspace.install_source
        def save(state):
            nonlocal armed
            original_save(state)
            current = state["episodes"]["1"]["results"].get("a4-coder", {})
            if armed and fault in {"publishing", "installing"} and current.get("phase") == fault:
                armed = False
                raise KeyboardInterrupt("synthetic result boundary loss")
        def install(workspace, content):
            nonlocal armed
            if armed and workspace == draft and fault in {"after_install", "partial"}:
                armed = False
                if fault == "partial":
                    (draft / "kernel.py").write_bytes(b"# interrupted native write")
                else:
                    original_install(workspace, content)
                raise KeyboardInterrupt("synthetic result install loss")
            return original_install(workspace, content)
        monkeypatch.setattr(lifecycle, "_save", save)
        monkeypatch.setattr(campaign.candidate_workspace, "install_source", install)
        if fault:
            with pytest.raises(KeyboardInterrupt):
                lifecycle.install_result(1, result, authority=authority)
            saved = fixed.run.load_state(NAMESPACE)["episodes"]["1"]["results"]["a4-coder"]
            with pytest.raises(ProtocolError, match="reconcile original"):
                lifecycle.run_episode(1)
            if fault == "partial":
                with pytest.raises(ProtocolError, match="retained last complete C"):
                    lifecycle.install_result(1, result, authority=authority)
                assert (draft / "kernel.py").read_bytes() == b"# interrupted native write"
                assert lifecycle.current_source(1) == last_complete
                assert fixed.run.load_state(NAMESPACE)["fence"]
                return
        binding = lifecycle.install_result(1, result, authority=authority)
        if fault:
            assert fixed.run.load_state(NAMESPACE)["episodes"]["1"]["results"]["a4-coder"]["publication_id"] == saved["publication_id"]
        expected = b"# \xce\xb1\r\ndef kernel():\r\n    return 7"
        assert (draft / "kernel.py").read_bytes() == expected
        assert fixed.run.bundle.source(binding.source) == expected
        assert binding.size == len(expected) and binding.kernel_sha256 == sha256(expected)
        assert binding.mode == "0644" and (draft / "kernel.py").stat().st_mode & 0o777 == 0o644
        assert result.invocation.terminal_usage.total_tokens == 9
        (draft / "kernel.py").write_bytes(expected + b"\n# later user progress")
        assert lifecycle.install_result(1, result, authority=authority) == binding
        assert (draft / "kernel.py").read_bytes().endswith(b"# later user progress")
        # Receipt settlement remains exclusively with A5, under this same owner.
        assert ledger.operations(fixed.run.run_id)["a4-coder"]["reservation"] == reservation
    finally:
        session.close()
        agent.close()


@pytest.mark.parametrize("option", ["always", "auto"])
def test_v1_is_explicitly_unsupported_not_free_optimization(fixed, tmp_path, option):
    campaign, engine, owner, lifecycle = engine_for(fixed, tmp_path)
    campaign.framework_baseline = option
    with pytest.raises(UnsupportedError, match="restricted V1"):
        lifecycle.run_episode(1)
    assert not campaign.evaluations and not campaign.agent.calls


def test_never_preserves_prefetch_and_native_metadata_helper(fixed, tmp_path, monkeypatch):
    campaign, engine, owner, lifecycle = engine_for(fixed, tmp_path, reviewers=True)
    calls = []
    def review():
        assert owner.events[0] == ("admit", "prepare", 1)
        calls.append("synthetic billed reviewer responsibility")
    monkeypatch.setattr(campaign, "_ensure_framework_baseline_correctness_guidance", review)
    lifecycle.run_episode(1)
    state = fixed.run.load_state(NAMESPACE)
    assert len(calls) == 1 and len(campaign.evaluations) == 1
    assert state["episodes"]["1"]["solution_metadata"] == {
        "present": False, "identity": None, "required": False}
    assert not (campaign.workspace / "framework_baseline.json").exists()


@pytest.mark.parametrize("damage", ["missing", "bytes", "mode", "extra", "metadata"])
def test_selected_public_seed_damage_never_falls_back(fixed, tmp_path, damage):
    campaign, engine, owner, lifecycle = engine_for(fixed, tmp_path)
    seed = lifecycle.current_source().source
    files = (fixed.run.root / seed["path"]).parent / "files"
    if damage == "missing":
        (files / "kernel.py").unlink()
    elif damage == "bytes":
        (files / "kernel.py").write_bytes(b"rewritten seed")
    elif damage == "mode":
        (files / "kernel.py").chmod(0o755)
    elif damage == "extra":
        (files / "extra.py").write_bytes(b"auxiliary source")
    else:
        (files / "solution.json").write_bytes(b"{}")
    with pytest.raises(ProtocolError):
        lifecycle.run_episode(1)
    assert not campaign.evaluations and not campaign.agent.calls


def test_empty_start_is_explicitly_unsupported(fixed):
    from aka.workflow.schema.source import selected_seed
    p = Publication(fixed.run.root)
    task = p.put_json("empty-start-task.json", dict(fixed.run.objects["task"], starter_source=None))
    p.commit([task])
    with pytest.raises(ProtocolError, match="empty_start"):
        selected_seed(fixed.run.bundle, task, None)
    assert selected_seed(fixed.run.bundle, task, fixed.task["starter"]) == fixed.task["starter"]


def test_native_evaluator_binding_cannot_change_under_the_same_seed(fixed, tmp_path):
    from aka.workflow.schema.task import BINDING
    campaign, engine, owner, lifecycle = engine_for(fixed, tmp_path)
    path = campaign.problem.evaluation.directory / BINDING
    binding = json.loads(path.read_bytes())
    binding["scope"]["sha256"] = "sha256:" + "0" * 64
    path.write_bytes(jcs(binding))
    with pytest.raises(ProtocolError):
        lifecycle.run_episode(1)
    assert not campaign.evaluations and not campaign.agent.calls


def test_original_v0_unknown_operation_can_remain_fenced(fixed, tmp_path):
    campaign, engine, owner, lifecycle = engine_for(fixed, tmp_path)
    campaign.evaluation_failure = KeyboardInterrupt("synthetic unknown V0 job")
    with pytest.raises(KeyboardInterrupt):
        lifecycle.run_episode(1)
    owner.denied.add("prepare")
    assert lifecycle.run_episode(1).reason == "external-admission: prepare"
    assert len(campaign.evaluations) == 1 and not campaign.agent.calls


def test_settled_v0_record_loss_cannot_repeat_evaluation(fixed, tmp_path, monkeypatch):
    campaign, engine, owner, lifecycle = pause_after_admission(fixed, tmp_path, monkeypatch)
    state = fixed.run.load_state(NAMESPACE)
    Path(state["v0"]["path"]).unlink()
    with pytest.raises(ProtocolError):
        lifecycle.run_episode(1)
    assert len(campaign.evaluations) == 1 and not campaign.agent.calls


def test_native_terminal_commit_loss_recovers_exact_c_without_new_call(fixed, tmp_path, monkeypatch):
    campaign, engine, owner, lifecycle = engine_for(fixed, tmp_path)
    original = campaign.candidate_workspace.record
    armed = True
    def record(workspace, **options):
        nonlocal armed
        result = original(workspace, **options)
        if armed and options["message"].startswith("v1: long-horizon episode"):
            armed = False
            raise KeyboardInterrupt("native terminal committed before return")
        return result
    monkeypatch.setattr(campaign.candidate_workspace, "record", record)
    with pytest.raises(KeyboardInterrupt):
        lifecycle.run_episode(1)
    state = fixed.run.load_state(NAMESPACE)
    draft = Path(state["episodes"]["1"]["draft"])
    expected = (draft / "kernel.py").read_bytes()
    outcome = SeedLifecycle.reopen(fixed.run, engine, owner).run_episode(1)
    assert outcome.attempt["recovered_after_supervisor_interruption"]
    assert fixed.run.bundle.source(lifecycle.current_source().source) == expected
    assert len(campaign.agent.calls) == len(campaign.evaluations) == 1


@pytest.mark.parametrize("fault", ["capture_intent", "published", "settled"])
def test_terminal_source_publication_replays_its_original_id(fixed, tmp_path, monkeypatch, fault):
    campaign, engine, owner, lifecycle = engine_for(fixed, tmp_path)
    original_save, original_freeze = lifecycle._save, lifecycle._freeze
    armed = True
    def save(state):
        nonlocal armed
        original_save(state)
        entry = state["episodes"].get("1", {})
        selected = (entry.get("terminal_publication_id") is not None and entry.get("terminal") is None
                    if fault == "capture_intent" else entry.get("terminal") is not None)
        if armed and fault != "published" and selected:
            armed = False
            raise KeyboardInterrupt("terminal source boundary loss")
    def freeze(workspace, **options):
        nonlocal armed
        result = original_freeze(workspace, **options)
        if armed and fault == "published" and options.get("publication_id"):
            armed = False
            raise KeyboardInterrupt("source published before terminal save")
        return result
    monkeypatch.setattr(lifecycle, "_save", save)
    monkeypatch.setattr(lifecycle, "_freeze", freeze)
    with pytest.raises(KeyboardInterrupt):
        lifecycle.run_episode(1)
    before = fixed.run.load_state(NAMESPACE)["episodes"]["1"]["terminal_publication_id"]
    expected = (Path(fixed.run.load_state(NAMESPACE)["episodes"]["1"]["draft"]) / "kernel.py").read_bytes()
    SeedLifecycle.reopen(fixed.run, engine, owner).run_episode(1)
    assert fixed.run.load_state(NAMESPACE)["episodes"]["1"]["terminal_publication_id"] == before
    assert fixed.run.bundle.source(lifecycle.current_source().source) == expected
    assert len(campaign.agent.calls) == len(campaign.evaluations) == 1
