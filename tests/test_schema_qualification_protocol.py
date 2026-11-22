"""Qualification protocol recovery through native admission, receipts and usage.

All model output is synthetic CPU data; no external model or GPU is invoked.
"""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aka.agent.provider import AgentOptions
from aka.contracts.workspace import RevisionContent
from aka.workflow.schema import materials, planner as module
from aka.workflow.schema.planner import Planner, PlannerConfig
from aka.workflow.schema.run_state import BudgetDenied, RunLimits, RunState
from test_schema_planner import CpuAgent, FIXTURE


@pytest.fixture
def review_harness(tmp_path, request):
    settings = dict(getattr(request, "param", {}))
    max_repairs = settings.pop("max_repairs", 1)
    state = RunState(tmp_path / "root", limits=RunLimits(continuous=True, **settings),
                     workspace=tmp_path / "native", identity={"synthetic-review": True})
    snapshot = materials.snapshot_inputs(tmp_path / "materials", ())
    ref = materials.publish_design(tmp_path / "design", FIXTURE, version=1,
        prepared_id="prepared", operation_id="import", materials_snapshot=snapshot, mode="import")
    actions, calls, usage = [], [], []
    def process(command, *, cwd, timeout, env):
        workspace = Path(cwd)
        calls.append({"workspace": workspace, "prompt": command[-1]})
        assert actions, "unexpected additional reviewer call"
        final = actions.pop(0)(workspace)
        if isinstance(final, tuple):
            return final
        return json.dumps({"type": "result", "result": final,
            "usage": {"input_tokens": 11, "output_tokens": 7}}), "", 0, False
    agent = CpuAgent(AgentOptions(state_directory=tmp_path / "agent"), process_runner=process,
        environment={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"})
    def observe(call_id, native):
        usage.append((call_id, native.session_id))
        state.record_usage(f"agent:{call_id}:{native.session_id}",
                           native.terminal_usage.total_tokens, native.terminal_usage.measurement)
    planner = Planner(agent, snapshot,
        config=PlannerConfig(timeout_seconds=7, max_repairs=max_repairs), observe_result=observe)
    kwargs = dict(design_ref=ref, prepared_id="prepared", operation_id="review-1",
        directory=tmp_path / "review", source=RevisionContent({"kernel.py": b"def copy(x):\n    return x\n"}),
        source_identity="sealed-candidate", evidence={"public_task": "Copy x to y.\n",
            "target_hardware": "Synthetic L20D target.\n", "seed_identity": "fixed-seed",
            "seed_files": {"kernel.py": b"pass\n"}})
    def run(**changes):
        return planner.qualify(**{**kwargs, **changes})
    try:
        with state.owned(), agent.invocation_scope(admit=state.admit_agent, observe=state.observe_agent):
            state.action("prepare")
            yield SimpleNamespace(planner=planner, state=state, actions=actions, calls=calls,
                                  usage=usage, kwargs=kwargs, run=run, directory=kwargs["directory"])
    finally:
        agent.close()


def review(workspace, *, verdict="pass", row_verdict=None, nested=False):
    binding = json.loads((workspace / "inputs/review/binding.json").read_text())
    candidate = {"source": "candidate/kernel.py", "start": 1, "end": 2, "claim": "direct scalar copy"}
    seed = {"source": "initial-seed/kernel.py", "start": 1, "end": 1, "claim": "previous no-op"}
    value = {"verdict": verdict, "design_identity": binding["design_identity"],
        "source_identity": binding["source_identity"], "rationale": "synthetic independent judgment",
        "obligations": [{"id": name, "verdict": row_verdict or verdict,
            "reason": "synthetic source inspection", "evidence": [candidate]} for name in binding["obligations"]],
        "substantive_change": True,
        "change_evidence": ([{"candidate": candidate, "seed": seed, "change": "implements copy"}]
                            if nested else [candidate, seed])}
    (workspace / "output/decision.json").write_text(json.dumps(value))
    return "reviewed"


def retained(directory):
    return {path.relative_to(directory).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("verdict,row_verdict,expected", [
    ("pass", "pass", "pass"), ("fail", "fail", "fail"), ("unknown", "unknown", "unknown"),
    ("pass", "fail", "fail"), ("pass", "unknown", "unknown")])
def test_well_formed_judgments_are_final(review_harness, verdict, row_verdict, expected):
    h = review_harness
    h.actions.append(lambda workspace: review(workspace, verdict=verdict, row_verdict=row_verdict))
    result = h.run()
    assert result["verdict"] == expected and result["qualified"] == (expected == "pass")
    assert result["calls"] == 1 and not result["diagnostics"]
    assert len(h.calls) == h.state.data["counts"]["agent_calls"] == 1


@pytest.mark.parametrize("verdict", ["pass", "fail", "unknown"])
def test_nested_change_evidence_gets_fresh_metered_correction(review_harness, verdict):
    h = review_harness
    h.actions.append(lambda workspace: review(workspace, verdict=verdict, nested=True))
    original = {}
    def correct(workspace):
        original.update(retained(h.directory / "attempts/1"))
        previous = json.loads((workspace / "inputs/previous-output/decision.json").read_text())
        assert previous["verdict"] == verdict and set(previous["change_evidence"][0]) == {"candidate", "seed", "change"}
        diagnostics = json.loads((workspace / "inputs/task/repair-diagnostics.json").read_text())
        assert "source/start/end/claim" in diagnostics[0]["message"]
        frozen = h.calls[0]["workspace"] / "inputs"
        for path in frozen.rglob("*"):
            if path.is_file():
                assert (workspace / "inputs" / path.relative_to(frozen)).read_bytes() == path.read_bytes()
        assert "real deviations remain fail" in h.calls[-1]["prompt"]
        assert "not a request for a passing verdict" in h.calls[-1]["prompt"]
        return review(workspace, verdict=verdict)
    h.actions.append(correct)
    result = h.run()
    assert result["verdict"] == verdict and result["qualified"] == (verdict == "pass")
    assert result["calls"] == 2 and result["diagnostics"]
    assert retained(h.directory / "attempts/1") == original
    assert len(h.calls) == h.state.data["counts"]["agent_calls"] == 2
    assert h.state.data["counts"]["tokens"] == 36
    assert [item[0] for item in h.usage] == ["review-1:attempt:1", "review-1:attempt:2"]
    assert h.usage[0][1] != h.usage[1][1]


@pytest.mark.parametrize("review_harness", [{"max_repairs": 0}, {"max_repairs": 1}], indirect=True)
def test_exhausted_format_error_stays_unknown_and_terminal(review_harness):
    h = review_harness
    limit = h.planner.config.max_repairs + 1
    h.actions.extend([lambda workspace: review(workspace, nested=True)] * limit)
    result = h.run()
    assert result["verdict"] == "unknown" and result["qualified"] is False
    assert result["substantive_change"] is False and result["calls"] == limit
    original = retained(h.directory)
    assert h.run() == result
    assert retained(h.directory) == original
    assert len(h.calls) == h.state.data["counts"]["agent_calls"] == limit
    assert h.state.data["counts"]["tokens"] == 18 * limit


@pytest.mark.parametrize("review_harness", [{"max_agent_calls": 1}, {"max_tokens": 18}], indirect=True)
def test_correction_cannot_bypass_root_budget(review_harness):
    h = review_harness
    h.actions.append(lambda workspace: review(workspace, nested=True))
    with pytest.raises(BudgetDenied):
        h.run()
    assert not (h.directory / "result.json").exists()
    result = h.run()
    assert result["verdict"] == "unknown" and result["qualified"] is False
    assert len(h.calls) == h.state.data["counts"]["agent_calls"] == 1
    assert h.state.data["counts"]["tokens"] == 18


@pytest.mark.parametrize("exit_status,timed_out", [(1, False), (0, True)])
def test_failed_invocation_is_not_an_output_correction(review_harness, exit_status, timed_out):
    h = review_harness
    def incomplete(workspace):
        review(workspace)
        return "reviewed", "", exit_status, timed_out
    h.actions.append(incomplete)
    result = h.run()
    assert result["verdict"] == "unknown" and result["calls"] == 1
    assert result["diagnostics"][0]["category"] == "invocation.failed"
    assert len(h.calls) == 1


@pytest.mark.parametrize("interruption", ["correction-receipt", "terminal-return"])
def test_lost_return_recovers_same_attempts_and_usage(review_harness, monkeypatch, interruption):
    h = review_harness
    h.actions.extend([lambda workspace: review(workspace, nested=True), review])
    if interruption == "correction-receipt":
        original = h.planner._collect
        def interrupted(attempt):
            if attempt.name == "2":
                raise RuntimeError("synthetic lost return")
            return original(attempt)
        owner, name = h.planner, "_collect"
    else:
        original = module.durable_write_json
        def interrupted(path, value, **kwargs):
            result = original(path, value, **kwargs)
            if path == h.directory / "result.json":
                raise RuntimeError("synthetic lost return")
            return result
        owner, name = module, "durable_write_json"
    monkeypatch.setattr(owner, name, interrupted)
    with pytest.raises(RuntimeError, match="synthetic lost return"):
        h.run()
    frozen = retained(h.directory / "attempts/1")
    intent = retained(h.directory)["intent.json"]
    monkeypatch.setattr(owner, name, original)
    result = h.run()
    assert result["qualified"] is True and result["calls"] == 2
    assert retained(h.directory / "attempts/1") == frozen
    assert retained(h.directory)["intent.json"] == intent
    assert len(h.calls) == h.state.data["counts"]["agent_calls"] == 2
    assert h.state.data["counts"]["tokens"] == 36


def test_unknown_correction_is_not_relaunched(review_harness):
    h = review_harness
    def lost(workspace):
        raise RuntimeError("synthetic missing original process receipt")
    h.actions.extend([lambda workspace: review(workspace, nested=True), lost])
    with pytest.raises(RuntimeError, match="missing original"):
        h.run()
    for _ in range(2):
        result = h.run()
        assert result["verdict"] == "unknown" and result["qualified"] is False
        assert result["diagnostics"][-1]["category"] == "invocation.unknown"
    assert not (h.directory / "result.json").exists()
    assert len(h.calls) == h.state.data["counts"]["agent_calls"] == 2
    assert h.state.data["counts"]["tokens"] == 18


@pytest.mark.parametrize("changed", ["task", "hardware", "source", "seed", "prompt", "repair-limit"])
def test_terminal_operation_rejects_changed_intent(review_harness, monkeypatch, tmp_path, changed):
    h = review_harness
    h.actions.append(review)
    h.run()
    original = retained(h.directory)
    changes = {}
    if changed in {"task", "hardware", "seed"}:
        key = {"task": "public_task", "hardware": "target_hardware", "seed": "seed_identity"}[changed]
        changes["evidence"] = {**h.kwargs["evidence"], key: "different"}
    elif changed == "source":
        changes["source"] = RevisionContent({"kernel.py": b"different source\n"})
    elif changed == "prompt":
        directory = tmp_path / "changed-prompts"
        directory.mkdir()
        (directory / "reviewer.md").write_text("different prompt")
        monkeypatch.setattr(module, "PROMPTS", directory)
    else:
        h.planner.config = replace(h.planner.config, max_repairs=2)
    with pytest.raises(ValueError, match="conflicts"):
        h.run(**changes)
    assert retained(h.directory) == original and len(h.calls) == 1


def test_damaged_retained_output_is_not_a_model_format_error(review_harness, monkeypatch):
    h = review_harness
    h.actions.append(lambda workspace: review(workspace, nested=True))
    original = h.planner._collect
    def interrupt(attempt):
        original(attempt)
        raise RuntimeError("synthetic interruption after collection")
    monkeypatch.setattr(h.planner, "_collect", interrupt)
    with pytest.raises(RuntimeError, match="after collection"):
        h.run()
    monkeypatch.setattr(h.planner, "_collect", original)
    path = h.directory / "attempts/1/raw/decision.json"
    path.chmod(0o600)
    path.write_text("corrupted retained bytes")
    with pytest.raises(ValueError, match="retained writer output changed"):
        h.run()
    assert len(h.calls) == h.state.data["counts"]["agent_calls"] == 1
    assert h.state.data["counts"]["tokens"] == 18
