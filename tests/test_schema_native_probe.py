"""Finite native capture through original owners; all model responses synthetic."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelschema.api import prepare_run
from kernelschema.artifacts import durable_write
from kernelschema.budget import OwnerAccount
from kernelschema.contracts import digest, jcs, loads
from kernelschema.runtime import ClaudeProfile
from aka.agent.backends import ClaudeAdapter
from aka.agent.runtime import terminal_usage_from_stream
from aka.workflow.schema import native_probe
from aka.workflow.schema.session import native_agent
from test_schema_optional_module import fixed, fake_runner


@pytest.fixture
def fixed_probe(fixed, tmp_path):
    return SimpleNamespace(run=prepare_run("implement", request=fixed.run.root / "request.json",
        out=tmp_path / "probe-run", backend_identity=native_probe.identity(fixed.run.objects["profile"])))


def probe_agent(fixed, trace, *, unknown=False, intervals=None, budget_exhausted=False, conflicting_usage=False):
    profile = ClaudeProfile(**fixed.run.objects["profile"]["parameters"]["claude"])
    def transform(events):
        answer = json.loads(events[-1]["result"])
        if answer["role"] == "reviewer":
            answer.update(kind="review", content={"transport_probe": True})
        events[-1]["result"] = json.dumps(answer)
        events[-1]["total_cost_usd"] = 0.0125
        if unknown:
            events[-1].pop("usage")
        if conflicting_usage or budget_exhausted:
            terminal = events[-1]
            terminal["usage"] = {key: 0 for key in terminal["usage"]}
            terminal["modelUsage"] = {profile.requested_model: {"inputTokens": 6, "outputTokens": 2219,
                "cacheCreationInputTokens": 112400, "cacheReadInputTokens": 0, "costUSD": 0.758005}}
            terminal["total_cost_usd"] = 0.758005
            if budget_exhausted:
                events.insert(-1, {"type": "assistant", "message": {"content": [
                    {"type": "text", "text": terminal["result"]}]}})
                terminal.update(subtype="error_max_budget_usd", is_error=True, result="",
                                errors=["Reached maximum budget ($0.5)"])
    original = fake_runner(trace, transform=transform)
    def run(command, *, cwd, timeout, env):
        if intervals is not None:
            intervals.append(timeout)
        intent = loads(Path(env["AKA_SCHEMA_OPERATION_INTENT"]).read_bytes())
        durable_write(Path(intent["native_locator"]["output_directory"]) / "exec-intent.json", jcs({
            "operation_id": intent["operation_id"], "session_id": intent["submission_key"],
            "locator_sha256": digest(intent["native_locator"]),
            "pid": 999999999, "boot_id": "synthetic CPU process only", "start_ticks": "1"}))
        stdout, stderr, status, timed_out = original(command, cwd=cwd, timeout=timeout, env=env)
        return stdout, stderr, 1 if budget_exhausted else status, timed_out
    return native_agent(profile, process_runner=run)


def test_probe_retains_two_original_calls_and_never_replays(fixed_probe, tmp_path):
    fixed = fixed_probe
    trace = []
    agent = probe_agent(fixed, trace)
    try:
        result = native_probe.capture(fixed.run, agent=agent)
        assert result["status"] == "completed" and result["synthetic"] is True
        assert result["usage"]["calls"]["coder"] == result["usage"]["calls"]["reviewer"] == 1
        assert result["usage"]["evaluator_jobs"] == 0 and result["usage"]["tokens"] == 18
        assert result["cli_reported_usd"] == "0.0250"
        assert not result["usage"]["unsettled"]
        value = loads(fixed.run.bundle.read(result["qualification"]))
        assert len(native_probe.references(value)) == 6
        changed = dict(value["roles"]["coder"])
        changed["input"] = value["roles"]["reviewer"]["input"]
        with pytest.raises(ValueError, match="locator mismatch"):
            native_probe.validate_capture(fixed.run.bundle, changed, agent.schema_adapter.profile, "coder",
                scope_timeout_ms=value["scope_timeout_ms"])
        again = native_probe.capture(fixed.run, agent=agent, continuing=True)
        assert again == result and len(trace) == 2
        owner = OwnerAccount.for_context(fixed.run, create=False)
        assert len(owner.operations(fixed.run.run_id)) == 2
        with pytest.raises(ValueError, match="actual completed"):
            native_probe.select_request(fixed.run.root / "request.json", fixed.run.root,
                tmp_path / "selected", run_root=tmp_path / "business")
        assert not (tmp_path / "selected").exists()
    finally:
        agent.close()


def test_probe_lost_return_reuses_original_receipt(fixed_probe, monkeypatch):
    fixed = fixed_probe
    from aka.workflow.schema.state import Transactions
    trace, fired = [], []
    original = Transactions.retain_invocation
    def interrupt(tx, operation, invocation):
        original(tx, operation, invocation)
        if not fired:
            fired.append(True)
            raise KeyboardInterrupt("native receipt retained before qualification return")
    monkeypatch.setattr(Transactions, "retain_invocation", interrupt)
    agent = probe_agent(fixed, trace)
    try:
        with pytest.raises(KeyboardInterrupt):
            native_probe.capture(fixed.run, agent=agent)
        result = native_probe.capture(fixed.run, agent=agent, continuing=True)
        assert result["status"] == "completed" and len(trace) == 2
        assert result["usage"]["calls"]["coder"] == result["usage"]["calls"]["reviewer"] == 1
        assert not result["usage"]["unsettled"]
    finally:
        agent.close()


def test_probe_unknown_usage_stops_without_signing_or_replacement(fixed_probe):
    fixed = fixed_probe
    trace = []
    agent = probe_agent(fixed, trace, unknown=True)
    try:
        result = native_probe.capture(fixed.run, agent=agent)
        assert result["status"] == "blocked" and result["qualification"] is None
        assert result["usage"]["calls"]["coder"] == 1
        assert result["usage"]["tokens"] is None
        assert native_probe.capture(fixed.run, agent=agent, continuing=True) == result
        assert len(trace) == 1
    finally:
        agent.close()


@pytest.mark.parametrize("budget_exhausted", [False, True])
def test_probe_conflicting_zero_usage_is_unknown_and_cannot_qualify(fixed_probe, budget_exhausted):
    fixed, trace = fixed_probe, []
    agent = probe_agent(fixed, trace, conflicting_usage=True, budget_exhausted=budget_exhausted)
    try:
        result = native_probe.capture(fixed.run, agent=agent)
        assert result["status"] == "blocked" and result["qualification"] is None
        assert result["usage"]["tokens"] is None and not result["usage"]["unsettled"]
        assert result["usage"]["calls"]["coder"] == 1 and result["usage"]["calls"]["reviewer"] == 0
        assert result["cli_reported_usd"] == "0.758005"
        state = fixed.run.load_state(native_probe.NAMESPACE)
        operation = next(iter(state["operations"].values()))
        invocation = loads((fixed.run.root / ".runtime/aka-native" / operation["id"] / "invocation.json").read_bytes())["invocation"]
        assert invocation["terminal_usage"]["total_tokens"] is None
        terminal = json.loads(invocation["stdout"].splitlines()[-1])
        assert all(value == 0 for value in terminal["usage"].values())
        assert terminal["modelUsage"]["synthetic-model"]["cacheCreationInputTokens"] == 112400
        if budget_exhausted:
            assert invocation["exit_status"] == 1 and not invocation["timed_out"]
            assert "native_terminal_unavailable" in result["reason"]
            assert '"type": "assistant"' in invocation["stdout"]
        assert native_probe.capture(fixed.run, agent=agent, continuing=True) == result and len(trace) == 1
    finally:
        agent.close()


# Parser-only B17 counterexamples; no native operation is launched.
ZERO = {"input_tokens": 0, "output_tokens": 0,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
MODEL = {"inputTokens": 6, "outputTokens": 2219, "cacheCreationInputTokens": 112400,
         "cacheReadInputTokens": 0, "costUSD": 0.758005}


@pytest.mark.parametrize("extra", [
    {"modelUsage": {"synthetic": MODEL}, "total_cost_usd": 0.758005},
    {"modelUsage": {"synthetic": {"inputTokens": 6}}},
    {"modelUsage": {"synthetic": {"costUSD": "0.758005"}}},
    {"total_cost_usd": 0.758005},
])
def test_zero_aggregate_conflict_never_falls_back_to_model_or_assistant_sum(extra):
    terminal = {"type": "result", "subtype": "error_max_budget_usd", "is_error": True,
                "usage": ZERO, **extra}
    # Assistant deltas are incomplete and may repeat; they cannot fix a terminal.
    assistant = {"type": "assistant", "message": {"id": "same", "usage": {"input_tokens": 112354}}}
    stream = "\n".join(json.dumps(event) for event in (assistant, assistant, terminal))
    events, usage = ClaudeAdapter().normalize_stream(stream)
    fallback = terminal_usage_from_stream(stream)
    assert usage.total_tokens is fallback.total_tokens is None
    assert usage.measurement == fallback.measurement == "unavailable"
    assert not any(event.kind == "terminal_usage" for event in events)
    assert json.loads(stream.splitlines()[-1]) == terminal


@pytest.mark.parametrize("events,total,measurement", [
    ([{"type": "result", "usage": ZERO, "total_cost_usd": 0,
       "modelUsage": {"synthetic": {"inputTokens": 0, "costUSD": 0}}}], 0, "exact"),
    ([{"type": "result", "usage": {**ZERO, "output_tokens": 4}, "total_cost_usd": 0.01}], 4, "exact"),
    ([{"type": "result", "modelUsage": {"synthetic": {"inputTokens": 6, "outputTokens": 4}}}], 10, "exact"),
    ([{"type": "assistant", "message": {"usage": {"input_tokens": 6}}}], 6, "partial"),
    ([{"type": "result"}], None, "unavailable"),
])
def test_nonconflicting_native_usage_keeps_existing_semantics(events, total, measurement):
    stream = "\n".join(json.dumps(event) for event in events)
    for usage in (ClaudeAdapter().normalize_stream(stream)[1], terminal_usage_from_stream(stream)):
        assert usage.total_tokens == total and usage.measurement == measurement
