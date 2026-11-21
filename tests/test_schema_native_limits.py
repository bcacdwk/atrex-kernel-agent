"""Declared business intervals and the separate finite qualification policy."""
from dataclasses import replace
from copy import deepcopy
import sys
from types import SimpleNamespace

import pytest

from kernelschema.api import open_run, prepare_run
from kernelschema.artifacts import Publication, durable_write
from kernelschema.budget import OwnerAccount
from kernelschema.contracts import ProtocolError, UnsupportedError, digest, jcs, loads, sha256
from kernelschema.runtime import ClaudeProfile
from aka.workflow.schema.calls import RoleCalls
from aka.workflow.schema.agate import native_admission
from aka.workflow.schema import native_probe
from aka.workflow.schema.native_probe import capture, identity, main, validate_capture
from aka.workflow.schema.session import native_agent
from aka.workflow.schema.state import Transactions
from test_schema_optional_module import fixed, fake_runner
from test_schema_native_probe import probe_agent


def extended_request(fixed, tmp_path, *, probe=False, scope_timeout_ms=None, profile_timeout_ms=240000,
                     request_active_ms=300000, native_active_ms=180000, executable=None, max_budget_usd="0.50"):
    profile = replace(ClaudeProfile(**fixed.run.objects["profile"]["parameters"]["claude"]),
                      timeout_ms=profile_timeout_ms, max_budget_usd=max_budget_usd)
    if executable is not None:
        profile = replace(profile, executable=str(executable), executable_sha256=sha256(executable.read_bytes()))
    limits = {"calls": {"coder": 1, "reviewer": 1, "planner": 0, "extract": 0},
              "model_calls": 2, "evaluator_jobs": 0, "active_ms": request_active_ms, "tokens": None}
    p = Publication(fixed.run.root)
    selected = {**fixed.run.objects["profile"], "parameters": {"claude": profile.as_dict()},
                "native_limits": {"workflow_limits": {**limits, "active_ms": native_active_ms}}}
    q = {**fixed.run.request, "request_id": "synthetic-extended-native-interval",
         "backend": {"name": "synthetic", "profile": p.put_json("extended-profile.json", selected)},
         "budget": {"parent": None, "limits": limits}}
    ref = p.put_json("extended-request.json", q)
    p.commit([ref])
    path = fixed.run.root / "extended-request.json"
    durable_write(path, jcs(q))
    binding = identity(selected, scope_timeout_ms=scope_timeout_ms) if probe else {"synthetic": "B10 declared native interval"}
    return prepare_run("implement", request=path, out=tmp_path / "extended-run", backend_identity=binding), profile


def test_probe_explicit_usd_threshold_keeps_exact_profile_and_finite_calls(fixed, tmp_path):
    run, profile = extended_request(fixed, tmp_path, probe=True, scope_timeout_ms=600000,
        profile_timeout_ms=600000, request_active_ms=1250000, native_active_ms=1250000, max_budget_usd="2.00")
    trace, intervals = [], []
    agent = probe_agent(SimpleNamespace(run=run), trace, intervals=intervals)
    try:
        result = capture(run, agent=agent)
        assert result["status"] == "completed" and result["synthetic"] is True
        assert len(trace) == len(intervals) == 2 and all(0 < t <= 600 for t in intervals)
        for _, command, _ in trace:
            assert command[command.index("--max-budget-usd") + 1] == "2.00"
        value = loads(run.bundle.read(result["qualification"]))
        assert value["claude_sha256"] == digest(profile.as_dict()) and value["scope_timeout_ms"] == 600000
        assert "aka.agent.backends.base" in value["implementation"]
        assert result["usage"]["calls"]["coder"] == result["usage"]["calls"]["reviewer"] == 1
        assert result["usage"]["evaluator_jobs"] == 0
        assert capture(run, agent=agent, continuing=True) == result and len(trace) == 2
        run.objects["profile"]["parameters"]["claude"]["max_budget_usd"] = "3.00"
        with pytest.raises(ProtocolError, match="scope/implementation binding"):
            capture(run, agent=agent, continuing=True)
        assert len(trace) == 2
    finally:
        agent.close()


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
    run, profile = extended_request(fixed, tmp_path, probe=True)
    trace, intervals = [], []
    agent = probe_agent(SimpleNamespace(run=run), trace, intervals=intervals)
    try:
        result = capture(run, agent=agent)
        assert result["status"] == "completed" and result["synthetic"] is True
        assert len(intervals) == 2 and all(0 < interval <= 120 for interval in intervals)
        qualification = loads(run.bundle.read(result["qualification"]))
        assert qualification["scope_timeout_ms"] == 120000
        assert qualification["claude_sha256"] == digest(profile.as_dict())
        assert profile.timeout_ms == 240000
        state = run.load_state("aka-native-qualification")
        assert all(op["maximum_timeout_ms"] == 120000 for op in state["operations"].values())
        assert all(role["locator"]["maximum_timeout_ms"] == 120000 for role in qualification["roles"].values())
        changed = {**qualification["roles"]["coder"],
                   "locator": {**qualification["roles"]["coder"]["locator"], "maximum_timeout_ms": 180000}}
        with pytest.raises(ProtocolError, match="locator mismatch"):
            validate_capture(run.bundle, changed, profile, "coder", scope_timeout_ms=120000)
        assert capture(run, agent=agent, continuing=True) == result and len(intervals) == 2
        tx = Transactions(run, state["binding"], continuing=True, namespace="aka-native-qualification")
        with pytest.raises(ProtocolError, match="original native call timeout ceiling changed"):
            RoleCalls(tx, agent).invoke("native-probe-coder", "coder", run.objects["task"]["starter_source"], maximum_timeout_ms=180000)
        assert len(intervals) == 2
    finally:
        agent.close()


@pytest.mark.parametrize("request_ms,native_ms,upper_seconds", [
    (400000, 400000, 300), (400000, 180000, 180), (90000, 400000, 90),
])
def test_explicit_probe_ceiling_reaches_native_process_under_original_limits(fixed, tmp_path, request_ms, native_ms, upper_seconds):
    run, profile = extended_request(fixed, tmp_path, probe=True, scope_timeout_ms=300000,
        profile_timeout_ms=360000, request_active_ms=request_ms, native_active_ms=native_ms)
    exact_profile = deepcopy(run.objects["profile"])
    exact_request = deepcopy(run.request)
    trace, intervals = [], []
    agent = probe_agent(SimpleNamespace(run=run), trace, intervals=intervals)
    try:
        result = capture(run, agent=agent)
        assert result["status"] == "completed" and len(intervals) == 2
        assert all(0 < interval <= upper_seconds for interval in intervals)
        if upper_seconds > 120:
            assert all(interval > 120 for interval in intervals)
        assert run.request == exact_request and run.objects["profile"] == exact_profile
        value = loads(run.bundle.read(result["qualification"]))
        assert value["scope_timeout_ms"] == run.anchor["implementation"]["scope_timeout_ms"] == 300000
        assert value["claude_sha256"] == digest(profile.as_dict())
        state = run.load_state(native_probe.NAMESPACE)
        assert state["binding"] == run.anchor["implementation"]
        assert all(op["maximum_timeout_ms"] == 300000 for op in state["operations"].values())
        for role, entry in value["roles"].items():
            assert entry["locator"]["maximum_timeout_ms"] == min(300000, request_ms)
            assert entry["locator"]["timeout_ceiling_ms"] == 300000
            validate_capture(run.bundle, entry, profile, role, scope_timeout_ms=300000)
            # Even when the actual request is tighter, a changed declared cap
            # cannot relabel the same execution as a different scope.
            for changed_cap in (180000, 360000):
                with pytest.raises(ProtocolError, match="locator mismatch"):
                    validate_capture(run.bundle, entry, profile, role, scope_timeout_ms=changed_cap)
            changed = deepcopy(entry)
            changed["locator"]["timeout_ceiling_ms"] = 360000
            with pytest.raises(ProtocolError, match="locator mismatch"):
                validate_capture(run.bundle, changed, profile, role, scope_timeout_ms=360000)
            changed = deepcopy(entry)
            changed["locator"]["maximum_timeout_ms"] -= 1
            with pytest.raises(ProtocolError, match="locator mismatch"):
                validate_capture(run.bundle, changed, profile, role, scope_timeout_ms=300000)
            publication = Publication(run.root)
            changed = deepcopy(entry)
            execution = loads(run.bundle.read(entry["execution"]))
            del execution["locator_sha256"]
            changed["execution"] = publication.put_bytes("missing-exec-binding.json", jcs(execution))
            publication.commit([changed["execution"]])
            with pytest.raises(ProtocolError, match="locator mismatch"):
                validate_capture(run.bundle, changed, profile, role, scope_timeout_ms=300000)
        # Negative admission checks remove the synthetic marker only to reach
        # the damaged evidence checks; these artifacts never qualify a run.
        for ceiling in (180000, 360000, None):
            bad = {**value, "synthetic": False, "scope_timeout_ms": ceiling}
            publication = Publication(run.root)
            ref = publication.put_bytes("invalid-qualification.json", jcs(bad))
            publication.commit([ref])
            materials = {"entries": [{"file": r, "roles": []} for r in [ref, *native_probe.references(value)]]}
            admission = SimpleNamespace(bundle=run.bundle, objects={"profile": run.objects["profile"], "materials": materials})
            with pytest.raises(ProtocolError, match="locator mismatch|timeout is missing"):
                native_admission(admission, ref)
            if ceiling == 180000:
                wrong_profile = deepcopy(run.objects["profile"])
                wrong_profile["parameters"]["claude"]["timeout_ms"] = 120000
                admission.objects["profile"] = wrong_profile
                with pytest.raises(UnsupportedError, match="bytes/profile"):
                    native_admission(admission, ref)
        assert capture(open_run("implement", run.root), agent=agent, continuing=True) == result
        assert len(intervals) == 2
    finally:
        agent.close()


def test_probe_bad_scope_parameters_rejected_before_preparation_or_calls(fixed, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid scope must not prepare a run, open an owner, or invoke an agent")
    monkeypatch.setattr(native_probe, "prepare_run", forbidden)
    monkeypatch.setattr(native_probe, "open_run", forbidden)
    monkeypatch.setattr(native_probe, "native_agent", forbidden)
    selected = fixed.run.objects["profile"]
    assert identity(selected)["scope_timeout_ms"] == 5000  # shorter declared profile
    for value in (True, False, 0, -1, 1.5, "1000", 5001, 9007199254740992):
        with pytest.raises(ProtocolError, match="positive safe integer"):
            identity(selected, scope_timeout_ms=value)
    for value in ("0", "-1", "5001", "9007199254740992"):
        with pytest.raises(ProtocolError, match="positive safe integer"):
            main(["--request", str(fixed.run.root / "request.json"), "--out", str(tmp_path / "bad"),
                  "--scope-timeout-ms", value])
    with pytest.raises(SystemExit) as rejected:
        main(["--continue", str(tmp_path / "absent"), "--scope-timeout-ms", "120000"])
    assert rejected.value.code == 2 and not (tmp_path / "bad").exists()


def test_cli_continue_preserves_long_scope_and_rejects_lost_bindings(fixed, tmp_path, monkeypatch, capsys):
    source, profile = extended_request(fixed, tmp_path, profile_timeout_ms=360000,
        request_active_ms=400000, native_active_ms=400000)
    root = tmp_path / "cli-probe"
    trace, intervals = [], []
    agent = probe_agent(SimpleNamespace(run=source), trace, intervals=intervals)
    monkeypatch.setattr(native_probe, "capture", lambda context, **options: capture(context, agent=agent, **options))
    retain = Transactions.retain_invocation
    def interrupt(tx, op, invocation):
        retain(tx, op, invocation)
        raise KeyboardInterrupt("original receipt retained")
    monkeypatch.setattr(Transactions, "retain_invocation", interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            main(["--request", str(source.root / "request.json"), "--out", str(root), "--scope-timeout-ms", "300000"])
        monkeypatch.setattr(Transactions, "retain_invocation", retain)
        assert len(intervals) == 1
        main(["--continue", str(root)])
        result = loads(capsys.readouterr().out)
        assert result["status"] == "completed" and len(intervals) == 2
        assert all(120 < interval <= 300 for interval in intervals)
        run = open_run("implement", root)
        original = run.load_state(native_probe.NAMESPACE)
        assert original["binding"]["scope_timeout_ms"] == 300000
        assert all(op["maximum_timeout_ms"] == 300000 for op in original["operations"].values())
        variants = []
        changed = deepcopy(original)
        changed["binding"]["scope_timeout_ms"] = 120000
        variants.append(changed)
        for field in ("binding", "probe"):
            changed = deepcopy(original)
            del changed[field]
            variants.append(changed)
        changed = deepcopy(original)
        del changed["operations"]["native-probe-coder"]["maximum_timeout_ms"]
        variants.append(changed)
        for changed in variants:
            run.save_state(native_probe.NAMESPACE, changed)
            with pytest.raises(ProtocolError, match="binding|records"):
                main(["--continue", str(root)])
        run.save_state(native_probe.NAMESPACE, original)
        state_path = root / f".runtime/state/{native_probe.NAMESPACE}.json"
        state_path.unlink()
        with pytest.raises(ProtocolError, match="state missing"):
            main(["--continue", str(root)])
        run.save_state(native_probe.NAMESPACE, original)
        anchor_path = root / ".runtime/run.json"
        anchor = loads(anchor_path.read_bytes())
        for changed in (120000, None):
            value = deepcopy(anchor)
            if changed is None:
                del value["implementation"]["scope_timeout_ms"]
            else:
                value["implementation"]["scope_timeout_ms"] = changed
            durable_write(anchor_path, jcs(value), replace=True)
            with pytest.raises(ProtocolError, match="binding"):
                main(["--continue", str(root)])
        durable_write(anchor_path, jcs(anchor), replace=True)
        main(["--continue", str(root)])
        assert loads(capsys.readouterr().out) == result and len(intervals) == 2
    finally:
        agent.close()


def test_probe_real_synthetic_child_retains_scope_locator_digest(fixed, tmp_path):
    """Run the actual exec shim and native process with a local synthetic CLI."""
    executable = tmp_path / "synthetic-native-cli"
    executable.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
payload = json.load(sys.stdin)
session = sys.argv[sys.argv.index("--session-id") + 1]
print(json.dumps({"type": "system", "subtype": "init", "tools": [], "mcp_servers": [],
    "skills": [], "plugins": [], "permissionMode": "dontAsk", "cwd": os.getcwd(),
    "session_id": session, "model": "synthetic-model"}))
answer = {"protocol": "ks.claude.output.v1", "role": payload["role"],
    "assignment_sha256": payload["assignment_sha256"], "status": "ok", "reason": "synthetic CPU child"}
if payload["role"] == "coder":
    answer.update(kind="candidate_source", content="# synthetic CPU child\\ndef kernel(): pass\\n")
else:
    answer.update(kind="review", content={"transport_probe": True})
print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
    "session_id": session, "result": json.dumps(answer), "total_cost_usd": 0,
    "usage": {"input_tokens": 5, "output_tokens": 4, "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0}}))
''')
    executable.chmod(0o755)
    run, profile = extended_request(fixed, tmp_path, probe=True, scope_timeout_ms=300000,
        profile_timeout_ms=360000, request_active_ms=400000, native_active_ms=400000, executable=executable)
    agent = native_agent(profile)
    try:
        result = capture(run, agent=agent)
        assert result["status"] == "completed" and result["synthetic"] is True
        value = loads(run.bundle.read(result["qualification"]))
        for role, entry in value["roles"].items():
            execution = loads(run.bundle.read(entry["execution"]))
            assert execution["locator_sha256"] == digest(entry["locator"])
            assert entry["locator"]["timeout_ceiling_ms"] == 300000
            assert entry["locator"]["maximum_timeout_ms"] == 300000
            validate_capture(run.bundle, entry, profile, role, scope_timeout_ms=300000)
    finally:
        agent.close()


def test_default_capture_never_recreates_lost_original_owner_or_scope(fixed, tmp_path, monkeypatch):
    run, _ = extended_request(fixed, tmp_path, probe=True)
    trace = []
    agent = probe_agent(SimpleNamespace(run=run), trace)
    try:
        result = capture(run, agent=agent)
        assert result["status"] == "completed" and len(trace) == 2
        state_path = run.root / f".runtime/state/{native_probe.NAMESPACE}.json"
        budget_path = run.root / ".runtime/state/budget.json"
        owner_path = run.root / ".runtime/owner-account.json"
        original = {path: path.read_bytes() for path in (state_path, budget_path, owner_path)}
        def forbidden(*args, **kwargs):
            pytest.fail("lost original facts cannot create or register any owner")
        monkeypatch.setattr(OwnerAccount, "create", forbidden)
        monkeypatch.setattr(OwnerAccount, "register", forbidden)
        # Public capture() defaults to continuing=False. Existing facts must
        # still select read-only missing-authority rejection, before register.
        for missing in ((budget_path,), (owner_path,), (budget_path, owner_path),
                        (state_path,), (state_path, budget_path, owner_path)):
            for path in missing:
                path.unlink()
            with pytest.raises(ProtocolError, match="binding missing|owner-account.json|state missing"):
                capture(run, agent=agent)
            assert all(not path.exists() for path in missing)
            assert len(trace) == 2
            for path in missing:
                durable_write(path, original[path])
        # Each existing local fact is enough even without any other retained
        # state, so loss of intents alone cannot turn recovery into creation.
        pristine = prepare_run("implement", request=run.root / "request.json", out=tmp_path / "gap-run",
            backend_identity=run.anchor["implementation"])
        for relative in (".runtime/state/budget.json", ".runtime/owner-account.json",
                         ".runtime/aka-native", ".runtime/terminals", "native-qualification-result.json"):
            marker = pristine.root / relative
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_bytes(b"original scope fact; never reconstruct an owner")
            with pytest.raises(ProtocolError, match="state missing"):
                capture(pristine, agent=agent)
            assert not (pristine.root / f".runtime/state/{native_probe.NAMESPACE}.json").exists()
            marker.unlink()
        assert len(trace) == 2
    finally:
        agent.close()
