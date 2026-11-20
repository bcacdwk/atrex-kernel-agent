"""Finite native capture through original owners; all model responses synthetic."""
import json
from pathlib import Path

import pytest

from kernelschema.artifacts import durable_write
from kernelschema.budget import OwnerAccount
from kernelschema.contracts import jcs, loads
from kernelschema.runtime import ClaudeProfile
from aka.workflow.schema import native_probe
from aka.workflow.schema.session import native_agent
from test_schema_optional_module import fixed, fake_runner


def probe_agent(fixed, trace, *, unknown=False):
    profile = ClaudeProfile(**fixed.run.objects["profile"]["parameters"]["claude"])
    def transform(events):
        answer = json.loads(events[-1]["result"])
        if answer["role"] == "reviewer":
            answer.update(kind="review", content={"transport_probe": True})
        events[-1]["result"] = json.dumps(answer)
        events[-1]["total_cost_usd"] = 0.0125
        if unknown:
            events[-1].pop("usage")
    original = fake_runner(trace, transform=transform)
    def run(command, *, cwd, timeout, env):
        intent = loads(Path(env["AKA_SCHEMA_OPERATION_INTENT"]).read_bytes())
        durable_write(Path(intent["native_locator"]["output_directory"]) / "exec-intent.json", jcs({
            "operation_id": intent["operation_id"], "session_id": intent["submission_key"],
            "pid": 999999999, "boot_id": "synthetic CPU process only", "start_ticks": "1"}))
        return original(command, cwd=cwd, timeout=timeout, env=env)
    return native_agent(profile, process_runner=run)


def test_probe_retains_two_original_calls_and_never_replays(fixed, tmp_path):
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
            native_probe.validate_capture(fixed.run.bundle, changed, agent.schema_adapter.profile, "coder")
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


def test_probe_lost_return_reuses_original_receipt(fixed, monkeypatch):
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


def test_probe_unknown_usage_stops_without_signing_or_replacement(fixed):
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
