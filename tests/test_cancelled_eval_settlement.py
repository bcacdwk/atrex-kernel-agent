"""CPU native launcher cancellation, original archive and read-only settlement."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import socket
from threading import Event, Thread
import time
from types import MethodType, SimpleNamespace

import pytest

from aka.contracts.execution import ExecutionPending, UnsupportedExecution
from aka.contracts.experiment import ExperimentRecordError
from aka.execution.gateway.jobs import GatewayJobs
from aka.experiment.runner import LegacyExperiment
from aka.workflow.schema.run_state import RecoveryRequired
from test_live_eval_recovery import CHILD, fixture, read_service


@pytest.fixture(autouse=True)
def only_loopback_network(monkeypatch):
    connect, getaddrinfo = socket.socket.connect, socket.getaddrinfo
    def local_connect(sock, address):
        assert address[0] == "127.0.0.1", "cancelled Eval fixture forbids nonloopback network"
        return connect(sock, address)
    def local_getaddrinfo(host, *args, **kwargs):
        assert host == "127.0.0.1", "cancelled Eval fixture forbids nonloopback DNS"
        return getaddrinfo(host, *args, **kwargs)
    monkeypatch.setattr(socket.socket, "connect", local_connect)
    monkeypatch.setattr(socket, "getaddrinfo", local_getaddrinfo)


def confirmation_http(monkeypatch, http):
    from aka.execution import live_eval
    def factory(environment, endpoint, *, transport_retries):
        assert transport_retries == 0
        return http
    monkeypatch.setattr(live_eval, "sdk_http", factory)


CANCEL_CHILD = CHILD.replace(
    'return 200, {"job_id": "cpu-original-job", "status": "cancelled" if state.get("cancelled") else "succeeded",',
    'return 200, {"job_id": "cpu-original-job", "status": "running" if state["mode"] == "cancel-pending" else "cancelled" if state.get("cancelled") else "succeeded",')


def cancelled(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch, mode="cancel-pending")
    (tmp_path / "cpu_launcher.py").write_text(CANCEL_CHILD)
    def cancel_owner():
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                if any(item["method"] == "GET" for item in read_service(f)["calls"]):
                    f.bound.cancel_event.set()
                    return
            except ValueError:
                pass
            time.sleep(.005)
    thread = Thread(target=cancel_owner)
    thread.start()
    with f.state.owned(), f.executor.dispatch_scope(admit=f.state.admit_execution, observe=f.state.observe_execution):
        f.state.action("implement")
        result = f.run()
    thread.join()
    assert result.transport_status == "cancelled" and result.facts.disposition == "unknown"
    assert len(result.attempts) == len(f.launches) == 2
    source = json.loads(f.command.submission)["request"]["candidate"].encode()
    f.records.publish_content(f.ref, source, "candidate-source")
    from aka.evaluation.measurement import gateway_identity
    identity = gateway_identity(json.loads(f.command.submission)["request"], {}, source_parameters={
        "candidate_source_format": "raw-kernel/1", "candidate_source_artifact_sha256": hashlib.sha256(source).hexdigest()})
    f.records._update(f.ref, lambda old: replace(old, steps=(replace(old.steps[0], measurement_identity=identity),)))
    f.records.failed(f.ref, ExecutionPending(result))
    f.experiment = LegacyExperiment(f.executor, records=f.records)
    f.gets = []
    f.job = {"job_id": "cpu-original-job", "kind": "eval", "status": "cancelled",
        "cleanup_pending": False, "execution_quarantined": False, "execution_state": "not_submitted"}
    def http(endpoint, method, path, payload, timeout):
        assert method == "GET" and path == "/v1/jobs/cpu-original-job?include=spec" and payload is None
        f.gets.append((method, path))
        return dict(f.job)
    confirmation_http(monkeypatch, http)
    f.original = (f.records._directory(f.ref) / "record.json").read_bytes()
    f.artifacts = {item.artifact_id: f.records.read_artifact(item) for item in f.records.get(f.ref).artifacts}
    return f


def test_cancelled_native_owner_settles_without_refund_replay_or_measurement(tmp_path, monkeypatch):
    f = cancelled(tmp_path, monkeypatch)
    counts = deepcopy(f.state.data["counts"])
    with f.state.owned(), pytest.raises(RecoveryRequired):
        f.state.reconcile(SimpleNamespace(query_job=lambda *a, **k: pytest.fail("native archive must precede Schema")))
    record = f.experiment.reconcile_cancelled_eval(f.ref)
    assert record.lifecycle == "terminal" and record.outcome == "inconclusive"
    assert record.evaluation is None and record.attempts[0].assessment is None
    result = record.attempts[0].result
    assert result.transport_status == "cancelled" and result.facts.disposition == "terminal"
    assert [item.disposition for item in result.attempts] == ["unknown", "unknown", "terminal"]
    assert [item.operation for item in result.attempts] == ["run", "run", "query"]
    assert result.attempts[-1].attempt_id is None
    assert f.state.data["counts"] == counts and len(f.launches) == 2 and len(f.gets) == 1
    assert all(f.records.read_artifact(item) == f.artifacts[item.artifact_id]
        for item in record.artifacts if item.artifact_id in f.artifacts)
    archived, = [f.records.read_artifact(item) for item in record.artifacts if item.role == "cancelled-eval-original-record"]
    assert archived == f.original
    assert f.experiment.reconcile_cancelled_eval(f.ref) == record and len(f.gets) == 1
    assert not any(item.disposition == "not_submitted" for item in record.attempts)


@pytest.mark.parametrize("field,value", [("status", "running"), ("status", "succeeded"),
    ("cleanup_pending", True), ("cleanup_pending", None), ("execution_quarantined", True),
    ("execution_quarantined", None), ("execution_state", "unknown"), ("job_id", "wrong-job"), ("kind", "dev"),
    ("idempotency_key", "wrong"), ("solution", {"source": {"type": "inline", "files": {"kernel.py": "wrong"}}})])
def test_uncertain_or_different_job_keeps_original_unknown_bytes(tmp_path, monkeypatch, field, value):
    f = cancelled(tmp_path, monkeypatch)
    f.job[field] = value
    with pytest.raises(ExperimentRecordError, match="experiment_pending"):
        f.experiment.reconcile_cancelled_eval(f.ref)
    assert (f.records._directory(f.ref) / "record.json").read_bytes() == f.original
    assert len(f.launches) == 2 and len(f.gets) == 1 and f.state.pending()


@pytest.mark.parametrize("change", [
    {"execution_state": "stopped", "remote_terminated": False},
    {"remote_terminated": "true"},
    {"error": {"details": {"cleanup_pending": True}}},
    {"error": {"details": {"execution_quarantined": True}}},
    {"remote_terminated": True, "error": {"details": {"cleanup_pending": True}}},
    {"remote_terminated": True, "error": {"details": {"execution_quarantined": True}}},
    {"error": {"details": {"cleanup_pending": None}}},
    {"error": {"details": {"execution_quarantined": None}}},
    {"error": {"details": {"execution_state": "running"}}},
    {"error": {"details": {"execution_state": "stopped"}}},
    {"error": {"details": {"remote_terminated": False}}},
    {"error": {"details": {"cause": {"details": {"cleanup_pending": True}}}}},
    {"execution_state": "running"},
    {"request_digest": {"gpu": "different-device"}},
    {"request_digest": {"num_gpus": 99}},
    {"request_digest": {"num_gpus": True}},
    {"request_digest": {"kind": "dev"}},
    {"request_digest": {"timeout_s": 99}},
    {"request_digest": "not-a-summary-object"},
    {"num_gpus": 99},
    {"mode": "smoke"},
])
def test_conflicting_confirmation_keeps_unknown_receipts(tmp_path, monkeypatch, change):
    f = cancelled(tmp_path, monkeypatch)
    f.job.update(change)
    with pytest.raises(ExperimentRecordError, match="experiment_pending"):
        f.experiment.reconcile_cancelled_eval(f.ref)
    assert (f.records._directory(f.ref) / "record.json").read_bytes() == f.original
    assert len(f.gets) == 1 and len(f.launches) == 2 and len(f.state.pending()) == 2
    assert all(f.records.read_artifact(item) == f.artifacts[item.artifact_id]
        for item in f.records.get(f.ref).artifacts)


@pytest.mark.parametrize("field", ["cleanup_pending", "execution_quarantined", "execution_state"])
def test_missing_cleanup_proof_keeps_original_bytes(tmp_path, monkeypatch, field):
    f = cancelled(tmp_path, monkeypatch)
    del f.job[field]
    with pytest.raises(ExperimentRecordError, match="experiment_pending"):
        f.experiment.reconcile_cancelled_eval(f.ref)
    assert (f.records._directory(f.ref) / "record.json").read_bytes() == f.original


@pytest.mark.parametrize("echo", [
    {"solution": None, "request_digest": None},
    {"request_digest": {"gpu": "NVIDIA L20D", "num_gpus": 1, "kind": "eval", "timeout_s": 60}},
    {"execution_state": "stopped", "remote_terminated": True,
     "error": {"details": {"cleanup_pending": False, "execution_quarantined": False, "execution_state": "stopped"}}},
])
def test_optional_or_matching_echo_and_confirmed_stop(tmp_path, monkeypatch, echo):
    f = cancelled(tmp_path, monkeypatch)
    f.job.update(echo)
    record = f.experiment.reconcile_cancelled_eval(f.ref)
    assert record.lifecycle == "terminal" and record.outcome == "inconclusive" and record.evaluation is None
    assert [part.disposition for part in record.attempts[0].result.attempts] == ["unknown", "unknown", "terminal"]


def test_operation_lock_precedes_confirmation(tmp_path, monkeypatch):
    f = cancelled(tmp_path, monkeypatch)
    with f.records.operation(f.records.get(f.ref).request_id):
        with pytest.raises(ExperimentRecordError, match="experiment_busy"):
            f.experiment.reconcile_cancelled_eval(f.ref)
    assert not f.gets and (f.records._directory(f.ref) / "record.json").read_bytes() == f.original


@pytest.mark.parametrize("response,other_origin", [
    *((status, other) for status in (301, 302, 303, 307, 308) for other in (False, True)),
    (503, False), (200, False),
])
def test_confirmation_network_is_one_bound_get_without_artifacts(monkeypatch, response, other_origin):
    from dataclasses import asdict
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from aka.contracts.execution_recovery import JobReference
    from aka.execution.gateway import jobs
    from aka.execution.job_recovery import GatewayRecoveryMixin
    from aka.execution.live_eval import digest, WIRE_PATH
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    def no_artifacts(*args, **kwargs):
        pytest.fail("cancelled confirmation must not collect artifacts")
    monkeypatch.setattr(jobs, "collect_job_files", no_artifacts)
    calls, forwarded = [], []
    class Target(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            forwarded.append(True)
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
    target = ThreadingHTTPServer(("127.0.0.1", 0), Target)
    destination = "http://127.0.0.1:" + str(target.server_port)
    class Origin(Target):
        def do_GET(self):
            calls.append((self.path, self.headers.get("Access-Key") == "loopback-only-ak",
                          bool(self.headers.get("Token"))))
            if len(calls) > 1:
                return super().do_GET()
            body = json.dumps({"job_id": "loopback-job", "kind": "eval", "status": "cancelled",
                "cleanup_pending": False, "execution_quarantined": False, "execution_state": "not_submitted",
                "result": {"artifacts": [{"name": "unrelated", "url": destination + "/artifact"}]}}).encode()
            self.send_response(response)
            if 300 <= response < 400:
                self.send_header("Location", (destination if other_origin else "") + "/redirected")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    origin = ThreadingHTTPServer(("127.0.0.1", 0), Origin)
    endpoint = "http://127.0.0.1:" + str(origin.server_port)
    servers = (origin, target)
    threads = [Thread(target=server.serve_forever, daemon=True) for server in servers]
    for thread in threads:
        thread.start()
    reference = JobReference("gateway", "loopback-job", endpoint=endpoint, operation="run",
        idempotency_key="loopback-only-key", request_identity="a" * 64)
    link = {"job_reference": asdict(reference), "idempotency_key": reference.idempotency_key,
        "request_identity": reference.request_identity, "route_identity": digest([endpoint, WIRE_PATH, "", ""]),
        "namespace_identity": digest("loopback-only-ak")}
    transport = GatewayJobs(endpoint=endpoint, environment={"AGATE_AK": "loopback-only-ak", "AGATE_SK": "loopback-only-sk"})
    class Executor(GatewayRecoveryMixin):
        def _job_transport(self):
            return transport
    try:
        result = Executor().query_recorded_eval(link, timeout=3)
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join()
    assert calls == [("/v1/jobs/loopback-job?include=spec", True, True)]
    assert not forwarded and not result.files and result.job_references == (reference,)
    assert len(result.attempts) == 1 and result.attempts[0].operation == "query"
    assert result.attempts[0].attempt_id is None
    if response == 200:
        assert result.transport_status == "cancelled" and result.facts.disposition == "terminal"
        assert json.loads(result.stdout)["result"]["artifacts"]
    else:
        assert result.transport_status == "infrastructure" and result.facts.disposition == "unknown"


@pytest.mark.parametrize("broken", ["namespace", "route", "key", "request", "source", "physical"])
def test_mismatched_retained_identity_never_queries(tmp_path, monkeypatch, broken):
    f = cancelled(tmp_path, monkeypatch)
    if broken == "namespace":
        f.transport.environment["AGATE_AK"] = "wrong-namespace"
    elif broken == "route":
        f.transport.endpoint = "https://wrong.invalid"
    elif broken == "source":
        f.records.publish_content(f.ref, b"# another source", "candidate-source")
    else:
        old = f.records.get(f.ref)
        attempt, = old.attempts
        result = attempt.result
        if broken == "physical":
            result = replace(result, attempts=(replace(result.attempts[0], attempt_id="wrong"), *result.attempts[1:]))
        else:
            key = "idempotency_key" if broken == "key" else "request_identity"
            result = replace(result, job_references=(replace(result.job_references[0], **{key: "wrong"}),))
        f.records._save(replace(old, attempts=(replace(attempt, result=result),)))
    before = (f.records._directory(f.ref) / "record.json").read_bytes()
    with pytest.raises((ExperimentRecordError, UnsupportedExecution)):
        f.experiment.reconcile_cancelled_eval(f.ref)
    assert not f.gets and (f.records._directory(f.ref) / "record.json").read_bytes() == before


@pytest.mark.parametrize("restart", ["none", "old-gate", "receipt", "wrong-task", "wrong-owner"])
def test_public_native_timeout_closes_both_owners_and_continues_next_episode(tmp_path, monkeypatch, restart):
    from aka.execution import live_eval
    from aka.contracts.execution_recovery import InfrastructureRetryPolicy
    from aka.execution.legacy.executor import LegacyExecutor
    from aka.workflow.light.long_horizon.store import CampaignStore
    from test_schema_prepared_rejection_resume import prepared_root
    monkeypatch.setattr(live_eval, "InfrastructureRetryPolicy", lambda: InfrastructureRetryPolicy((0, 0), 120))
    with prepared_root(tmp_path, monkeypatch, replay=True, mode="accepted", probe=False) as f:
        from aka.workflow.light.orchestration.campaign import Campaign
        from aka.workflow.light.campaign import ManagedCampaign
        f.campaign.register_runtime_episode = MethodType(Campaign.register_runtime_episode, f.campaign)
        f.campaign.agent_access = MethodType(ManagedCampaign.agent_access, f.campaign)
        f.campaign.module_options.episode_timeout_seconds = 1
        environment_backend = LegacyExecutor._environment_backend
        def backend(executor):
            value = environment_backend(executor)
            value.backend, value.endpoint = "legacy", "cpu-fixture"
            check = value.check
            def observed(*args, **kwargs):
                support = check(*args, **kwargs)
                return replace(support, environment=replace(support.environment, source="observed"))
            value.check = observed
            return value
        monkeypatch.setattr(LegacyExecutor, "_environment_backend", backend)
        assert f.driver.run(f.campaign) == "interrupted: requested"
        baseline, = f.campaign.experiment_records()
        f.driver = f.reopen()
        # Baseline is complete before injecting an actual owned launcher cancel.
        original_process = f.campaign.agent._process_runner
        original_run = LegacyExecutor.run
        coder_calls, gets, threads = [], [], []
        phase = {"cancel": False}
        def run(executor, command):
            if phase["cancel"] and command.submission is not None:
                cancel = Event()
                command = replace(command, cancel_event=cancel)
                def expire():
                    end = time.monotonic() + 10
                    while time.monotonic() < end:
                        try:
                            value = json.loads(f.state_path.read_text())
                            if any(item["method"] == "GET" and item["path"].startswith("/v1/jobs/cpu-original-job")
                                   for item in value["calls"]):
                                cancel.set()
                                return
                        except ValueError:
                            pass
                        time.sleep(.005)
                thread = Thread(target=expire)
                threads.append(thread)
                thread.start()
            return original_run(executor, command)
        monkeypatch.setattr(LegacyExecutor, "run", run)
        def process(command, *, cwd, timeout, env):
            if (Path(cwd) / "inputs/task/request.json").is_file() or coder_calls:
                return original_process(command, cwd=cwd, timeout=timeout, env=env)
            coder_calls.append(str(cwd))
            source = Path(cwd) / "kernel.py"
            source.write_bytes(source.read_bytes() + b"\n# CPU draft before timeout\n")
            state = json.loads(f.state_path.read_text())
            state.update(mode="cancel-pending", jobs={})
            f.state_path.write_text(json.dumps(state))
            child = tmp_path / "cpu-launcher.py"
            imports = child.read_text().split("from pathlib import Path", 1)[0]
            body = CANCEL_CHILD.replace('state_path = Path(os.environ["CPU_STATE"])',
                "state_path = Path(" + repr(str(f.state_path)) + ")").replace(
                'len(posts) == 1', 'len([p for p in posts if p.get("payload") and p["payload"].get("idempotency_key") == key]) == 1')
            child.write_text(imports + body)
            phase["cancel"] = True
            try:
                f.campaign.measure_for_acceptance(Path(cwd), ["--kind", "run", "--mode", "full", "--no-sync"])
            except ExecutionPending:
                pass
            finally:
                phase["cancel"] = False
                for thread in threads:
                    thread.join()
            return "CPU native Coder budget expired", "", -9, True
        f.campaign.agent._process_runner = process
        def http(endpoint, method, path, payload, timeout):
            assert method == "GET" and path == "/v1/jobs/cpu-original-job?include=spec" and payload is None
            gets.append((method, path))
            return {"job_id": "cpu-original-job", "kind": "eval", "status": "cancelled",
                "cleanup_pending": False, "execution_quarantined": False, "execution_state": "not_submitted"}
        transport = GatewayJobs(endpoint="https://cpu.invalid", http=http,
            environment={"AGATE_AK": "cpu-ak", "AGATE_SK": "cpu-sk"}, idempotent_replay=True)
        monkeypatch.setattr(LegacyExecutor, "_job_transport", lambda self: transport)
        confirmation_http(monkeypatch, http)
        f.driver.stop_event = None
        before_settle = []
        reconcile = f.driver._reconcile_cancelled_episode
        def checkpoint(campaign, **kwargs):
            if kwargs.get("facts") and not before_settle:
                facts = kwargs["facts"]
                before_settle.append((deepcopy(f.driver.state.data), facts["records"][-1]))
                if restart == "old-gate":
                    return False  # Exercise the old controller's actual failure gate.
                if restart == "wrong-owner":
                    next(iter(f.driver.state.pending().values()))["owner"]["timeout"] += 1
                elif restart == "wrong-task":
                    task = campaign.problem.evaluation
                    from aka.contracts.content import WorkspaceFile
                    campaign.problem.evaluation = replace(task, files=tuple(
                        WorkspaceFile(item.path, text="# different Task") if item.path == "reference.py" else item
                        for item in task.files))
            return reconcile(campaign, **kwargs)
        monkeypatch.setattr(f.driver, "_reconcile_cancelled_episode", checkpoint)
        native_reconcile = LegacyExperiment.reconcile_cancelled_eval
        receipts = []
        def lost_return(experiment, *args, **kwargs):
            result = native_reconcile(experiment, *args, **kwargs)
            if not receipts:
                receipts.append(result)
                raise OSError("CPU lost native settlement return")
            return result
        if restart == "receipt":
            monkeypatch.setattr(LegacyExperiment, "reconcile_cancelled_eval", lost_return)
        if restart in {"wrong-task", "wrong-owner"}:
            with pytest.raises(RecoveryRequired, match="original Task or admission owner"):
                f.driver.run(f.campaign)
            assert not gets and f.campaign.experiment_records()[-1].lifecycle == "failed"
            assert len(f.driver.state.pending()) == 2
            return
        if restart in {"old-gate", "receipt"}:
            error = RecoveryRequired if restart == "old-gate" else OSError
            message = "Original Experiment owner is unresolved" if restart == "old-gate" else "CPU lost native settlement return"
            with pytest.raises(error, match=message):
                f.driver.run(f.campaign)
            checkpoint_state = deepcopy(f.driver.state.data)
            assert checkpoint_state["binding"]["episode"] == 1 and len(f.driver.state.pending()) == 2
            f.driver = f.reopen()
            f.driver.stop_event = None
            f.driver.resume_after_repair("CPU cancelled original Eval; exact archived Episode", campaign=f.campaign)
            repaired = deepcopy(f.driver.state.data["counts"])
            f.driver.resume_after_repair("CPU repeat native receipt", campaign=f.campaign)
            assert f.driver.state.data["counts"] == repaired
        assert f.driver.run(f.campaign) == "budget: episodes"
        data = f.driver.state.data
        assert data["counts"]["settled_episodes"] == 2 and data["binding"] is None and not f.driver.state.pending()
        state, old = before_settle[0]
        assert old.lifecycle == "failed" and old.attempts[0].disposition == "unknown"
        timeout_call, = [call for call in state["calls"].values() if call.get("timed_out")]
        after_timeout, = [call for call in data["calls"].values() if call.get("timed_out")]
        assert timeout_call == after_timeout and timeout_call["active_seconds"] > 0
        assert data["counts"]["dispatches"] == state["counts"]["dispatches"] == len(f.launches) == 3
        records = f.campaign.experiment_records()
        assert records[0] == baseline and records[-1].lifecycle == "terminal"
        assert records[-1].outcome == "inconclusive" and records[-1].evaluation is None
        native, active = CampaignStore.checkpoint(f.campaign.workspace)
        assert active is None and native.episodes == 2
        assert all(not item["accepted"] and item["candidate_commit"] is None for item in native.attempts)
        assert len(gets) == 1 and len(coder_calls) == 1
        assert f.campaign.setup_count == 1
