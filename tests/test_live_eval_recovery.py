"""Real CPU launcher/SDK calls, original FileExperimentRecords and Schema permits."""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from threading import Event
from types import SimpleNamespace

import pytest

from aka.contracts.execution_environment import EnvironmentQuery
from aka.contracts.execution_recovery import InfrastructureRetryPolicy
from aka.execution.environment_query import _snapshot, match_environment
from aka.execution.legacy.executor import LegacyExecutor
from aka.execution.legacy.runtime import SandboxRequest, ExecutionHooks
from aka.execution.processes import spawn_owned_session
from aka.experiment.file_records import FileExperimentRecords
from aka.experiment.runner import LegacyExperiment
from aka.workflow.schema.run_state import BudgetDenied, RecoveryRequired, RunLimits, RunState
from test_eval_submission_identity import command_at, owner_at


CHILD = r'''
import json, os, sys
from pathlib import Path
assert sys.dont_write_bytecode
def audit(event, args):
    if event.startswith("socket.") and event in {"socket.connect", "socket.getaddrinfo"}:
        raise AssertionError("CPU fixture forbids network")
sys.addaudithook(audit)
from atrex_gateway_client import Client
from atrex_gateway_client.errors import GatewayError
from aka.execution.gateway import sandbox
state_path = Path(os.environ["CPU_STATE"])
def request(self, method, path, data, timeout, attempt, **kwargs):
    state = json.loads(state_path.read_text())
    payload = json.loads(data) if data else None
    state["calls"].append({"method": method, "path": path, "payload": payload,
        "endpoint": self.base_url, "ak": self.auth.access_key})
    if method == "POST" and path == "/v1/jobs/eval":
        posts = [item for item in state["calls"] if item["method"] == "POST"]
        key = payload["idempotency_key"]
        if len(posts) == 1 and state["mode"] not in {"accepted", "get-failure", "cancel-get"}:
            if state["mode"] == "accepted-loss":
                state["jobs"][key] = "cpu-original-job"
            state_path.write_text(json.dumps(state))
            error = {"message": "<html>503 tengine original evidence</html>", "trace_id": "cpu-original-trace",
                "extra_original_field": {"do_not_drop": [1, "full original receipt"]}}
            if state["mode"] == "policy-never":
                error["replay_policy"] = "never"
            elif state["mode"] == "execution-unknown":
                error["execution_state"] = "unknown"
            elif state["mode"] == "invalid-fact-channel":
                from aka.execution.outcomes import emit_fact
                emit_fact({"event": "invalid"})
            raise GatewayError(503, error)
        if state["mode"] in {"400", "401", "409", "404", "persistent"}:
            state_path.write_text(json.dumps(state))
            raise GatewayError(503 if state["mode"] == "persistent" else int(state["mode"]), {})
        state["jobs"].setdefault(key, "cpu-original-job")
        state_path.write_text(json.dumps(state))
        return 200, {"job_id": state["jobs"][key], "status": "queued", "kind": "eval"}
    state_path.write_text(json.dumps(state))
    if method == "GET":
        assert path.startswith("/v1/jobs/cpu-original-job")
        if state["mode"] == "get-failure":
            raise GatewayError(401, {"message": "CPU original-job GET denied"})
        if state["mode"] == "cancel-get" and not state.get("cancelled"):
            raise KeyboardInterrupt("CPU owner cancelled during original-job wait")
        return 200, {"job_id": "cpu-original-job", "status": "cancelled" if state.get("cancelled") else "succeeded",
            "result": {"exit_code": 0}}
    assert method == "POST" and path == "/v1/jobs/cpu-original-job/cancel"
    state["cancelled"] = True
    state_path.write_text(json.dumps(state))
    return 200, {"job_id": "cpu-original-job", "status": "cancelled"}
Client._request_once = request
sandbox._find_agate = lambda: "/cpu/agate"
# The ordinary off/unrecorded path still invokes the real SDK CLI. Substitute
# only that executable boundary; it shares the same fake native HTTP service.
import contextlib, io, subprocess
from atrex_gateway_client import cli
from atrex_gateway_client.auth import AkSkAuth
cli._client = lambda args: Client(args.url, auth=AkSkAuth("cpu-ak", "cpu-sk"), timeout=1, transport_retries=0)
cli._update.maybe_warn_outdated = lambda version: None
def cpu_cli(argv, **kwargs):
    assert argv[0] == "/cpu/agate"
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            cli.main(argv[1:])
        except SystemExit as stopped:
            code = stopped.code
    return subprocess.CompletedProcess(argv, code, stdout.getvalue(), stderr.getvalue())
sandbox.subprocess.run = cpu_cli
raise SystemExit(sandbox.main())
'''


def fixture(tmp_path, monkeypatch, *, mode="not-accepted", abba=False, limit=3, enabled=True,
            command_factory=command_at, child_source=CHILD):
    # Import is deliberate: an unavailable SDK is a failed test, never a skip.
    import atrex_gateway_client
    from aka.execution import live_eval
    monkeypatch.setattr(live_eval, "InfrastructureRetryPolicy", lambda: InfrastructureRetryPolicy((0, 0), 120))
    child = tmp_path / "cpu_launcher.py"
    child.write_text(child_source)
    state_path = tmp_path / "cpu-service.json"
    state_path.write_text(json.dumps({"mode": mode, "calls": [], "jobs": {}}))
    environment = {"PATH": "/usr/bin:/bin", "AGATE_AK": "cpu-ak", "AGATE_SK": "cpu-sk",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CPU_STATE": str(state_path), "ATREX_SANDBOX_QUEUE_WAIT_GRACE": "0",
        "PYTHONPATH": os.pathsep.join((str(Path(__file__).resolve().parents[1] / "src"),
            str(Path(atrex_gateway_client.__file__).resolve().parents[1])))}
    command = replace(command_factory(tmp_path, abba=abba), environment=environment, cancel_event=Event())
    records, ref, attempt = owner_at(tmp_path, command)
    bound = LegacyExperiment._observed_command(command, records, ref, attempt)
    request = SandboxRequest(tmp_path, (sys.executable, str(child)), "L20D", "", "https://cpu.invalid",
        60, (), environment)
    launches = []
    def spawn(argv, **kwargs):
        launches.append(argv)
        return spawn_owned_session(argv, **kwargs)
    executor = LegacyExecutor(request, ExecutionHooks(spawn, lambda: None, lambda env: 0),
        configuration={"url": "https://cpu.invalid", "idempotent_replay": enabled})
    transport = SimpleNamespace(endpoint="https://cpu.invalid", profile="", environment=dict(environment),
        idempotent_replay=enabled)
    monkeypatch.setattr(executor, "_job_transport", lambda: transport)
    fields = json.loads((Path(__file__).parent / "fixtures/agate_b41_environment.json").read_text())
    snapshot = _snapshot("legacy", "cpu-fixture", EnvironmentQuery("L20D"), fields["detail"],
        fields["capabilities"], [], LegacyExecutor.capabilities.operations)
    support = match_environment(EnvironmentQuery("L20D", command.operation), snapshot)
    monkeypatch.setattr(executor, "_environment_backend", lambda: SimpleNamespace(check=lambda *a, **k: support))
    state = RunState(tmp_path / "schema", limits=RunLimits(max_dispatches=limit), workspace=tmp_path, identity={})
    def run():
        try:
            result = executor.run(bound)
        except BaseException as error:
            records.observed(ref, attempt, error=error)
            raise
        records.observed(ref, attempt, result=result)
        return result
    return SimpleNamespace(executor=executor, transport=transport, records=records, ref=ref, attempt=attempt,
        bound=bound, command=command, state=state, state_path=state_path, launches=launches, run=run)


def read_service(f):
    return json.loads(f.state_path.read_text())


@pytest.mark.parametrize("abba", [False, True])
@pytest.mark.parametrize("mode", ["not-accepted", "accepted-loss"])
def test_same_live_owner_recovers_and_settles_both_ledgers(tmp_path, monkeypatch, abba, mode):
    f = fixture(tmp_path, monkeypatch, mode=mode, abba=abba)
    with f.state.owned(), f.executor.dispatch_scope(admit=f.state.admit_execution, observe=f.state.observe_execution):
        f.state.action("implement")
        f.state.data["window"] = {"start_counts": dict(f.state.data["counts"])}
        result = f.run()
        assert result.facts.disposition == "terminal", result.as_dict()
        assert len(f.launches) == f.state.data["counts"]["dispatches"] == 2
        assert len(result.attempts) == 2
        assert len({item.attempt_id for item in result.attempts}) == 2
        assert all(item.disposition == "terminal" for item in result.attempts)
        assert all(call["state"] == "settled" for call in f.state.data["calls"].values())
        assert len({call["original_permit"] for call in f.state.data["calls"].values()}) == 1
        assert all(call["live_eval"]["attempt_id"] == f.attempt for call in f.state.data["calls"].values())
    service = read_service(f)
    posts = [call for call in service["calls"] if call["method"] == "POST"]
    assert len(posts) == 2 and posts[0]["payload"] == posts[1]["payload"]
    assert posts[0]["payload"]["options"]["timeout_s"] == 60
    assert len(service["jobs"]) == 1
    assert all(call["path"].startswith("/v1/jobs/") for call in service["calls"])
    fresh = FileExperimentRecords(tmp_path / "records", "cpu", create=False)
    record = fresh.get(f.ref)
    assert len(record.attempts) == 1 and record.attempts[0].attempt_id == f.attempt
    assert record.attempts[0].disposition == "terminal"
    physical = [json.loads(fresh.read_artifact(item)) for item in record.artifacts if item.role == "live_eval_physical"]
    assert len(physical) == 2 and physical[0]["result"]["facts"]["disposition"] == "unknown"
    assert "503 tengine original evidence" in json.dumps(physical[0])
    assert "cpu-original-trace" in json.dumps(physical[0])
    assert "extra_original_field" in json.dumps(physical[0])
    assert "503 tengine original evidence" in json.dumps(result.as_dict())
    assert all(not (fresh.root / f.ref.experiment_id / "artifacts" / item.artifact_id).stat().st_mode & 0o222
        for item in record.artifacts if item.role.startswith("live_eval_"))


@pytest.mark.parametrize("mode", ["400", "401", "409", "404", "persistent"])
def test_later_rejection_or_exhaustion_cannot_erase_first_unknown(tmp_path, monkeypatch, mode):
    f = fixture(tmp_path, monkeypatch, mode=mode)
    with f.state.owned(), f.executor.dispatch_scope(admit=f.state.admit_execution, observe=f.state.observe_execution):
        f.state.action("prepare")
        result = f.run()
        assert result.facts.disposition == "unknown"
        assert len(f.launches) == f.state.data["counts"]["dispatches"] == (3 if mode == "persistent" else 2)
        assert all(call["state"] == "unknown" for call in f.state.data["calls"].values())
        assert f.records.get(f.ref).attempts[0].disposition == "unknown"
    assert not read_service(f)["jobs"]
    assert all(call["path"] == "/v1/jobs/eval" for call in read_service(f)["calls"])


@pytest.mark.parametrize("stop", ["quota", "window", "other-unknown", "cancel", "deadline", "auth", "route", "body", "lower-limit"])
def test_no_new_native_dispatch_after_revocation(tmp_path, monkeypatch, stop):
    f = fixture(tmp_path, monkeypatch, limit=1 if stop == "quota" else 3)
    original = f.executor._run_sandbox
    def physical(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(f.launches) == 1:
            if stop == "window":
                f.state.limits = replace(f.state.limits, window_dispatches=1)
            elif stop == "other-unknown":
                f.state.data["calls"]["different-owner"] = {"kind": "execution", "state": "unknown"}
            elif stop == "cancel":
                f.bound.cancel_event.set()
            elif stop == "deadline":
                kwargs["continuation"].deadline = 0
            elif stop == "auth":
                f.transport.environment["AGATE_AK"] = "another-ak"
            elif stop == "route":
                f.executor.configuration["url"] = "https://changed.invalid"
            elif stop == "body":
                object.__setattr__(f.bound, "submission", f.bound.submission + b" ")
            elif stop == "lower-limit":
                f.state.limits = replace(f.state.limits, per_job_timeout_seconds=1)
        return result
    monkeypatch.setattr(f.executor, "_run_sandbox", physical)
    with f.state.owned(), f.executor.dispatch_scope(admit=f.state.admit_execution, observe=f.state.observe_execution):
        f.state.action("implement")
        f.state.data["window"] = {"start_counts": dict(f.state.data["counts"])}
        with pytest.raises(Exception):
            f.run()
        assert len(f.launches) == f.state.data["counts"]["dispatches"] == 1
        assert f.records.get(f.ref).attempts[0].disposition == "unknown"
        assert f.state.pending()
    assert len(read_service(f)["calls"]) == 1


@pytest.mark.parametrize("prior_503", [False, True])
def test_known_job_get_failure_never_posts_again(tmp_path, monkeypatch, prior_503):
    f = fixture(tmp_path, monkeypatch, mode="not-accepted" if prior_503 else "get-failure")
    if prior_503:
        original = f.executor._run_sandbox
        def physical(*args, **kwargs):
            result = original(*args, **kwargs)
            value = read_service(f)
            value["mode"] = "get-failure"
            f.state_path.write_text(json.dumps(value))
            return result
        monkeypatch.setattr(f.executor, "_run_sandbox", physical)
    with f.state.owned(), f.executor.dispatch_scope(admit=f.state.admit_execution, observe=f.state.observe_execution):
        f.state.action("prepare")
        result = f.run()
        assert result.facts.disposition == "unknown" and result.job_references
        assert f.state.data["counts"]["dispatches"] == 1 + prior_503
        assert all(call["job_references"] for call in f.state.pending().values())
    assert sum(call["method"] == "POST" for call in read_service(f)["calls"]) == 1 + prior_503


@pytest.mark.parametrize("condition", ["gate-off", "scope-off", "unrecorded-key", "missing-key"])
def test_normal_and_ineligible_paths_do_not_gain_live_retries(tmp_path, monkeypatch, condition):
    from contextlib import nullcontext
    f = fixture(tmp_path, monkeypatch, enabled=condition != "gate-off")
    if condition in {"unrecorded-key", "missing-key"}:
        f.bound = replace(f.bound, on_observation=None,
            idempotency_key=None if condition == "missing-key" else f.bound.idempotency_key)
        if condition == "missing-key":
            prepared = json.loads(f.bound.submission)
            prepared["reference_files"] = {"reference.py": "# reference", "input.py": "# inputs",
                "shapes.json": json.dumps({"0": {"N": 32}})}
            f.bound = replace(f.bound, submission=json.dumps(prepared).encode())
    with f.state.owned():
        f.state.action("prepare")
        scope = f.executor.dispatch_scope(admit=f.state.admit_execution,
            observe=f.state.observe_execution) if condition != "scope-off" else nullcontext()
        with scope:
            result = f.executor.run(f.bound)
            assert result.facts.disposition == "unknown"
            assert len(f.launches) == 1
            assert f.state.data["counts"]["dispatches"] == (0 if condition == "scope-off" else 1)
            if condition != "scope-off":
                with pytest.raises(RuntimeError, match="replay"):
                    f.executor.replay_job(f.bound, idempotency_key="forbidden")
    assert len(read_service(f)["calls"]) == 1, result.as_dict()


def test_restart_and_closed_scope_cannot_recreate_live_permission(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch, mode="persistent")
    with f.state.owned(), f.executor.dispatch_scope(admit=f.state.admit_execution, observe=f.state.observe_execution):
        f.state.action("prepare")
        assert f.run().facts.disposition == "unknown"
        stale = f.executor.bind_sandbox(f.executor.request, f.executor.hooks)
    with pytest.raises(RuntimeError, match="closed"):
        stale.run(f.bound)
    before = len(read_service(f)["calls"])
    fresh = FileExperimentRecords(tmp_path / "records", "cpu", create=False)
    rebuilt = LegacyExperiment._observed_command(f.command, fresh, f.ref, f.attempt)
    assert not hasattr(rebuilt.on_observation, "_live_eval_owner")
    with f.state.owned(), f.executor.dispatch_scope(admit=f.state.admit_execution, observe=f.state.observe_execution):
        with pytest.raises(RecoveryRequired):
            f.executor.run(rebuilt)
    assert len(read_service(f)["calls"]) == before


def test_default_bound_and_sdk_cancel_share_original_auth_route(tmp_path, monkeypatch):
    assert InfrastructureRetryPolicy().delays == (5, 10)
    assert InfrastructureRetryPolicy().max_elapsed_s == 120
    f = fixture(tmp_path, monkeypatch, mode="cancel-get")
    with f.state.owned(), f.executor.dispatch_scope(admit=f.state.admit_execution, observe=f.state.observe_execution):
        f.state.action("prepare")
        result = f.run()
        assert result.facts.cancellation_requested
        assert f.state.data["counts"]["dispatches"] == 1
    calls = read_service(f)["calls"]
    assert any(call["path"] == "/v1/jobs/cpu-original-job/cancel" for call in calls)
    assert all(call["ak"] == "cpu-ak" and call["endpoint"] == "https://cpu.invalid" for call in calls)


def test_original_step_deadline_interrupts_the_fixed_retry_delay(tmp_path, monkeypatch):
    import time
    from aka.execution import live_eval
    from aka.experiment.step_budget import BudgetCancellation
    f = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(live_eval, "InfrastructureRetryPolicy", InfrastructureRetryPolicy)
    # Use the real default 5-second delay with an original owner clock that
    # expires during that delay; the production cancellation wrapper does the wait.
    budget = BudgetCancellation(float("inf"))
    object.__setattr__(f.bound, "cancel_event", budget)
    original = f.executor._run_sandbox
    def physical(*args, **kwargs):
        result = original(*args, **kwargs)
        budget.deadline = time.time() + .02
        return result
    monkeypatch.setattr(f.executor, "_run_sandbox", physical)
    with f.state.owned(), f.executor.dispatch_scope(admit=f.state.admit_execution, observe=f.state.observe_execution):
        f.state.action("prepare")
        from concurrent.futures import CancelledError
        with pytest.raises(CancelledError):
            f.run()
        assert f.state.data["counts"]["dispatches"] == 1 and f.state.pending()
    assert len(read_service(f)["calls"]) == 1


def test_sdk_adapter_refuses_redirect_without_a_second_http_call(monkeypatch):
    import io
    import urllib.error
    from email.message import Message
    from aka.execution.live_eval import sdk_http
    from aka.execution.gateway.sandbox import GatewayHTTPError
    calls = []
    # Exercise the SDK's real urllib request and redirect handler. Fake only
    # the network handler's response; no socket or external service is used.
    import urllib.request
    class Redirect(urllib.request.HTTPSHandler):
        def https_open(self, request):
            calls.append(request.full_url)
            headers = Message()
            headers["Location"] = "https://changed.invalid/v1/jobs/eval"
            response = urllib.response.addinfourl(io.BytesIO(b"redirect"), headers, request.full_url, 302)
            response.msg = "Found"
            return response
    real_build = urllib.request.build_opener
    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: real_build(Redirect(), *handlers))
    request = sdk_http({"AGATE_AK": "cpu-ak", "AGATE_SK": "cpu-sk"}, "https://cpu.invalid")
    with pytest.raises(GatewayHTTPError) as error:
        request("https://cpu.invalid", "POST", "/v1/jobs/eval", {"idempotency_key": "cpu-key"}, 1)
    assert error.value.status == 302
    assert calls == ["https://cpu.invalid/v1/jobs/eval"]


@pytest.mark.parametrize("mode", ["policy-never", "execution-unknown", "invalid-fact-channel"])
def test_original_service_policy_veto_is_preserved(tmp_path, monkeypatch, mode):
    f = fixture(tmp_path, monkeypatch, mode=mode)
    with f.state.owned(), f.executor.dispatch_scope(admit=f.state.admit_execution, observe=f.state.observe_execution):
        f.state.action("prepare")
        result = f.run()
        assert result.facts.disposition == "unknown" and f.state.pending()
        assert f.state.data["counts"]["dispatches"] == 1
    assert len(read_service(f)["calls"]) == 1


@pytest.mark.parametrize("failure", ["archive", "observation"])
def test_failed_second_receipt_finalization_preserves_current_job_and_attempt(tmp_path, monkeypatch, failure):
    f = fixture(tmp_path, monkeypatch)
    if failure == "archive":
        publish = f.records.publish_content
        physical = []
        def archive(ref, content, role):
            if role == "live_eval_physical":
                physical.append(role)
                if len(physical) == 2:
                    raise OSError("CPU failed second physical archive")
            return publish(ref, content, role)
        monkeypatch.setattr(f.records, "publish_content", archive)
    else:
        observe = f.records.observed
        injected = []
        def observation(ref, attempt_id, result=None, error=None):
            if (not injected and result is not None and result.facts.local_stopped
                    and len(result.attempts) == 2):
                injected.append(True)
                raise OSError("CPU failed final owner observation")
            return observe(ref, attempt_id, result=result, error=error)
        monkeypatch.setattr(f.records, "observed", observation)
    with f.state.owned(), f.executor.dispatch_scope(admit=f.state.admit_execution, observe=f.state.observe_execution):
        f.state.action("prepare")
        with pytest.raises(OSError) as raised:
            f.run()
        result = raised.value.execution_outcome
        assert len(result.attempts) == 2 and len(result.job_references) == 1
        record = f.records.get(f.ref)
        assert len(record.attempts) == 1 and len(record.attempts[0].result.attempts) == 2
        assert record.attempts[0].result.job_references == result.job_references
        assert f.state.data["counts"]["dispatches"] == 2
        assert result.facts.disposition == ("unknown" if failure == "archive" else "terminal")
    assert len(read_service(f)["jobs"]) == 1


@pytest.mark.parametrize("slow", ["admission", "spawn"])
def test_absolute_client_deadline_includes_admission_and_spawn(tmp_path, monkeypatch, slow):
    import time
    f = fixture(tmp_path, monkeypatch)
    def admit(**request):
        permit = f.state.admit_execution(**request)
        request["continuation"].deadline = time.monotonic() + .03
        if slow == "admission":
            time.sleep(.06)
        return permit
    if slow == "spawn":
        original = f.executor.hooks.spawn
        def spawn(*args, **kwargs):
            time.sleep(.06)
            return original(*args, **kwargs)
        f.executor.hooks = replace(f.executor.hooks, spawn=spawn)
    with f.state.owned(), f.executor.dispatch_scope(admit=admit, observe=f.state.observe_execution):
        f.state.action("prepare")
        if slow == "admission":
            with pytest.raises(TimeoutError):
                f.run()
        else:
            result = f.run()
            assert result.facts.cancellation_requested
        assert f.state.data["counts"]["dispatches"] == 1
    assert read_service(f)["calls"] == []


def test_cancellation_protocol_only_needs_is_set(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    object.__setattr__(f.bound, "cancel_event", SimpleNamespace(is_set=lambda: False))
    with f.state.owned(), f.executor.dispatch_scope(admit=f.state.admit_execution, observe=f.state.observe_execution):
        f.state.action("prepare")
        assert f.run().facts.disposition == "terminal"
        assert f.state.data["counts"]["dispatches"] == 2


def test_expired_submit_deadline_keeps_bounded_original_job_cleanup(tmp_path, monkeypatch):
    import time
    from atrex_gateway_client import Client
    from aka.execution.live_eval import sdk_http
    from aka.execution.gateway.sandbox import GatewayHTTPError
    calls = []
    def wire(self, method, path, data, *args, **kwargs):
        calls.append((method, path, self.auth.access_key, self.base_url))
        return 200, {"job_id": "cpu-original-job", "status": "cancelled"}
    monkeypatch.setattr(Client, "_request_once", wire)
    request = sdk_http({"AGATE_AK": "cpu-ak", "AGATE_SK": "cpu-sk"}, "https://cpu.invalid",
        deadline=time.monotonic() - 1)
    with pytest.raises(GatewayHTTPError) as error:
        request("https://cpu.invalid", "POST", "/v1/jobs/eval", {"idempotency_key": "cpu-key"}, 1)
    assert error.value.status == 408 and not calls
    request("https://cpu.invalid", "POST", "/v1/jobs/cpu-original-job/cancel", {}, 1)
    request("https://cpu.invalid", "GET", "/v1/jobs/cpu-original-job", None, 1)
    assert len(calls) == 2 and all(item[2:] == ("cpu-ak", "https://cpu.invalid") for item in calls)


def test_archival_time_spends_the_same_recovery_elapsed_allowance(tmp_path, monkeypatch):
    import time
    from aka.execution import live_eval
    f = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(live_eval, "InfrastructureRetryPolicy", lambda: InfrastructureRetryPolicy((0, 0), .05))
    publish = f.records.publish_content
    def archive(ref, content, role):
        result = publish(ref, content, role)
        if role == "live_eval_physical":
            time.sleep(.06)
        return result
    monkeypatch.setattr(f.records, "publish_content", archive)
    with f.state.owned(), f.executor.dispatch_scope(admit=f.state.admit_execution, observe=f.state.observe_execution):
        f.state.action("prepare")
        with pytest.raises(TimeoutError):
            f.run()
        assert f.state.data["counts"]["dispatches"] == 1 and f.state.pending()
    assert len(read_service(f)["calls"]) == 1
