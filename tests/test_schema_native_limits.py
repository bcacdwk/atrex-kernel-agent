"""Declared business intervals and the separate finite qualification policy."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kernelschema.api import prepare_run
from kernelschema.artifacts import Publication, durable_write
from kernelschema.contracts import ProtocolError, digest, jcs, loads
from kernelschema.runtime import ClaudeProfile
from aka.workflow.schema.calls import RoleCalls
from aka.workflow.schema.native_probe import capture, validate_capture
from aka.workflow.schema.session import native_agent
from aka.workflow.schema.state import Transactions
from test_schema_optional_module import fixed, fake_runner
from test_schema_native_probe import probe_agent


def extended_request(fixed, tmp_path):
    profile = replace(ClaudeProfile(**fixed.run.objects["profile"]["parameters"]["claude"]), timeout_ms=240000)
    limits = {"calls": {"coder": 1, "reviewer": 1, "planner": 0, "extract": 0},
              "model_calls": 2, "evaluator_jobs": 0, "active_ms": 300000, "tokens": None}
    p = Publication(fixed.run.root)
    selected = {**fixed.run.objects["profile"], "parameters": {"claude": profile.as_dict()},
                "native_limits": {"workflow_limits": {**limits, "active_ms": 180000}}}
    q = {**fixed.run.request, "request_id": "synthetic-extended-native-interval",
         "backend": {"name": "synthetic", "profile": p.put_json("extended-profile.json", selected)},
         "budget": {"parent": None, "limits": limits}}
    ref = p.put_json("extended-request.json", q)
    p.commit([ref])
    path = fixed.run.root / "extended-request.json"
    durable_write(path, jcs(q))
    return prepare_run("implement", request=path, out=tmp_path / "extended-run",
                       backend_identity={"synthetic": "B10 declared native interval"}), profile


def test_business_native_session_uses_declared_timeout_and_original_owner(fixed, tmp_path):
    run, profile = extended_request(fixed, tmp_path)
    trace, intervals = [], []
    original = fake_runner(trace)
    def process(command, *, cwd, timeout, env):
        intervals.append(timeout)
        return original(command, cwd=cwd, timeout=timeout, env=env)
    agent = native_agent(profile, process_runner=process)
    try:
        tx = Transactions(run, {"synthetic": "B10 explicit native interval"})
        result, _, operation, _ = RoleCalls(tx, agent).invoke("long-business-coder", "coder", run.objects["task"]["starter_source"])
        assert result.output is not None and result.permission_errors == ()
        assert len(intervals) == 1 and 120 < intervals[0] <= 180
        assert operation["reservation"]["active_ms"] <= 180000
        owned = tx.owner.operations(run.run_id)[operation["id"]]
        assert owned["terminated"] and len(owned["history"]) == 1
        assert owned["latest"]["calls"] == 1 and owned["latest"]["tokens"] == 9
    finally:
        agent.close()


def test_native_probe_caps_actual_calls_without_changing_long_profile(fixed, tmp_path):
    run, profile = extended_request(fixed, tmp_path)
    trace, intervals = [], []
    agent = probe_agent(SimpleNamespace(run=run), trace, intervals=intervals)
    try:
        result = capture(run, agent=agent)
        assert result["status"] == "completed" and result["synthetic"] is True
        assert len(intervals) == 2 and all(0 < interval <= 120 for interval in intervals)
        qualification = loads(run.bundle.read(result["qualification"]))
        assert qualification["claude_sha256"] == digest(profile.as_dict())
        assert profile.timeout_ms == 240000
        state = run.load_state("aka-native-qualification")
        assert all(op["maximum_timeout_ms"] == 120000 for op in state["operations"].values())
        assert all(role["locator"]["maximum_timeout_ms"] == 120000 for role in qualification["roles"].values())
        changed = {**qualification["roles"]["coder"],
                   "locator": {**qualification["roles"]["coder"]["locator"], "maximum_timeout_ms": 180000}}
        with pytest.raises(ProtocolError, match="locator mismatch"):
            validate_capture(run.bundle, changed, profile, "coder")
        assert capture(run, agent=agent, continuing=True) == result and len(intervals) == 2
        tx = Transactions(run, state["binding"], continuing=True, namespace="aka-native-qualification")
        with pytest.raises(ProtocolError, match="original native call timeout ceiling changed"):
            RoleCalls(tx, agent).invoke("native-probe-coder", "coder", run.objects["task"]["starter_source"], maximum_timeout_ms=180000)
        assert len(intervals) == 2
    finally:
        agent.close()
