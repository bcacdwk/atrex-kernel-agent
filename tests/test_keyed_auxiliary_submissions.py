"""Owned native tools keep one immutable submission across bounded admission loss."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from aka.contracts.experiment import ExperimentRecordError
from aka.execution.live_eval import digest
from aka.experiment.file_records import FileExperimentRecords
from aka.experiment.runner import LegacyExperiment
from aka.workflow.schema.run_state import RecoveryRequired
from aka.workflow.schema.workflow import SchemaWorkflow
from test_eval_submission_identity import command_at as eval_command, owner_at
from test_live_eval_recovery import CHILD, fixture as live_fixture, read_service


KINDS = ("profile", "check", "disassemble")
PATHS = {"profile": "/v1/jobs/profile", "check": "/v1/jobs/compile", "disassemble": "/v1/jobs/disassemble"}


def command_at(path, kind):
    original = eval_command(path)
    prepared = json.loads(original.submission)
    prepared["kind"] = kind
    # All fields are synthetic. Include local-only/Eval controls so a raw
    # submission cannot accidentally forward the obsolete full Eval envelope.
    prepared["request"].update(init_kwargs={"N": 32}, timeout_s=60,
        arch="sm_90", sanitize="memcheck", fmt="ptx", level="deep", profiler="ncu",
        counters=["cpu.counter"], launch_skip=2, launch_count=3, kernel_regex="cpu_kernel")
    prepared["reference_files"] = {"reference.py": "# CPU reference", "input.py": "# CPU input",
                                   "shapes.json": '{"0":{"N":32}}'}
    return replace(original, operation=kind, submission=json.dumps(prepared).encode())


DURABLE_CHECK = '''
    if method == "POST" and not path.endswith("/cancel"):
        from aka.contracts.experiment import ExperimentRef
        from aka.experiment.file_records import FileExperimentRecords
        from aka.execution.live_eval import digest
        records = FileExperimentRecords(state_path.parent / "records", "cpu", create=False)
        record = records.get(ExperimentRef(**state["ref"]))
        intent, = [json.loads(records.read_artifact(a)) for a in record.artifacts if a.role == "submission_intent"]
        assert intent["idempotency_key"] == payload["idempotency_key"]
        assert any(a.sha256 == intent["submission_sha256"] for a in record.artifacts)
        if state["expect_live"]:
            frozen, = [json.loads(records.read_artifact(a)) for a in record.artifacts if a.role == "live_eval_submission"]
            assert frozen["request"] == payload
            root = json.loads(Path(state["schema_path"]).read_text())
            owner = list(root["calls"].values())[-1]
            assert owner["live_eval"] == frozen["identity"]
            assert owner["live_eval"]["request_identity"] == digest(payload)
            assert owner["live_eval"]["route_identity"] == digest([self.base_url, path, "", ""])
            assert owner["live_eval"]["namespace_identity"] == digest(self.auth.access_key)
        state.setdefault("wire_bytes", []).append(data.decode())
'''

ERROR_MODES = '''
        if state["mode"] in {"immediate-400", "overload", "transport"}:
            state_path.write_text(json.dumps(state))
            if state["mode"] == "transport":
                from atrex_gateway_client.errors import TransportError
                raise TransportError(cause_class="ConnectionResetError", stage="read", errno=104, attempts=1, retryable=True)
            if state["mode"] == "overload":
                raise GatewayError(429, {"reason":"gateway_overloaded", "details":{"retry_after_s":0.001}})
            raise GatewayError(400, {"reason":"invalid_request"})
'''


def child_source():
    child = CHILD.replace('path == "/v1/jobs/eval"', 'path in ' + repr(set(PATHS.values())))
    child = child.replace('{"accepted", "get-failure", "cancel-get"}',
                          '{"accepted", "get-failure", "cancel-get", "cancel-get-unconfirmed"}')
    child = child.replace('state["mode"] == "cancel-get"',
                          'state["mode"] in {"cancel-get", "cancel-get-unconfirmed"}')
    child = child.replace('state["mode"] == "get-failure"',
        '(state["mode"] == "get-failure" or state["mode"] == "cancel-get-unconfirmed" and state.get("cancelled"))')
    child = child.replace('    payload = json.loads(data) if data else None\n',
                          '    payload = json.loads(data) if data else None\n' + DURABLE_CHECK)
    child = child.replace('        key = payload["idempotency_key"]\n',
                          '        key = payload["idempotency_key"]\n' + ERROR_MODES)
    child = child.replace('                state["jobs"][key] = "cpu-original-job"',
        '                state["executions"] = state.get("executions", 0) + int(key not in state["jobs"])\n'
        '                state["jobs"][key] = "cpu-original-job"')
    child = child.replace('        state["jobs"].setdefault(key, "cpu-original-job")',
        '        state["executions"] = state.get("executions", 0) + int(key not in state["jobs"])\n'
        '        state["jobs"].setdefault(key, "cpu-original-job")')
    return child


def fixture(tmp_path, monkeypatch, kind, *, command_factory=None, **options):
    f = live_fixture(tmp_path, monkeypatch,
        command_factory=command_factory or (lambda path, **kwargs: command_at(path, kind)), child_source=child_source(), **options)
    service = read_service(f)
    service.update(ref=f.ref.as_dict(), schema_path=str(f.state.path), expect_live=options.get("enabled", True))
    f.state_path.write_text(json.dumps(service))
    return f


@pytest.mark.parametrize("explicit", [False, True])
def test_public_profile_options_are_frozen_without_losing_prepared_field_precedence(tmp_path, monkeypatch, explicit):
    from aka.contracts.execution import ProfileOptions
    def command(path, **kwargs):
        value = command_at(path, "profile")
        prepared = json.loads(value.submission)
        for field in ("level", "profiler", "counters", "top_kernels"):
            prepared["request"].pop(field, None)
        options = (ProfileOptions("deep", "ncu", ("cpu.public.counter",), "must-not-override-prepared", 7)
                   if explicit else ProfileOptions(level="deep"))
        return replace(value, submission=json.dumps(prepared).encode(), profile_options=options)
    f = fixture(tmp_path, monkeypatch, "profile", command_factory=command)
    assert run(f).facts.disposition == "terminal"
    wire = read_service(f)["calls"][0]["payload"]
    assert wire["level"] == "deep"
    if explicit:
        assert wire["profiler"] == "ncu" and wire["counters"] == ["cpu.public.counter"] and wire["top_kernels"] == 7
    else:
        assert not {"profiler", "counters", "top_kernels"}.intersection(wire)
    assert wire["kernel_regex"] == "cpu_kernel"


def test_final_body_publication_spends_original_client_deadline(tmp_path, monkeypatch):
    import time
    f = fixture(tmp_path, monkeypatch, "check")
    f.executor.request = replace(f.executor.request, wall_timeout=.02)
    publish = f.records.publish_content
    def archive(ref, content, role):
        saved = publish(ref, content, role)
        if role == "live_eval_submission":
            time.sleep(.04)
        return saved
    monkeypatch.setattr(f.records, "publish_content", archive)
    f.executor.hooks = replace(f.executor.hooks, spawn=lambda *a, **k: pytest.fail("expired owner cannot spawn"))
    with pytest.raises(TimeoutError):
        run(f)
    assert f.state.data["counts"]["dispatches"] == 0 and read_service(f)["calls"] == []


def run(f):
    with f.state.owned(), f.executor.dispatch_scope(admit=f.state.admit_execution, observe=f.state.observe_execution):
        f.state.action("implement")
        return f.run()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode", ["not-accepted", "accepted-loss", "accepted"])
def test_public_dispatch_freezes_wire_before_post_and_settles_same_remote_execution(tmp_path, monkeypatch, kind, mode):
    f = fixture(tmp_path, monkeypatch, kind, mode=mode)
    result = run(f)
    service = read_service(f)
    posts = [c for c in service["calls"] if c["method"] == "POST"]
    expected_count = 1 if mode == "accepted" else 2
    assert len(posts) == len(f.launches) == f.state.data["counts"]["dispatches"] == expected_count
    assert service["executions"] == 1
    assert len(set(service["wire_bytes"])) == 1
    assert all(c["path"] == PATHS[kind] for c in posts)
    assert all(c["ak"] == "cpu-ak" and c["endpoint"] == "https://cpu.invalid" for c in service["calls"])
    wire = posts[0]["payload"]
    assert wire["idempotency_key"] == f.bound.idempotency_key
    original = json.loads(f.command.submission)["request"]
    for field in ("candidate", "spec", "requirements", "deps_mode", "env_vars"):
        assert wire[field] == original[field]
    assert "mode" not in wire and "runner_overrides" not in wire
    if kind == "profile":
        assert wire["reference"] == original["reference"]
        assert wire["options"] == {"timeout_s": 60, "bench_iters": 100}
        assert all(wire[k] == original[k] for k in ("level", "profiler", "counters", "launch_skip", "launch_count", "kernel_regex"))
        assert "timeout_s" not in wire and "fmt" not in wire
        assert any(b.scope == "job" and b.effective == 60 for b in result.budgets)
    else:
        assert "timeout_s" not in wire and "options" not in wire and "reference" not in wire
        assert wire["init_kwargs"] == {"N": 32}
        assert not any(b.scope == "job" for b in result.budgets)
        assert any(d.category == "observation" and d.code == "job-runtime-budget-unconfirmed" for d in result.diagnostics)
        if kind == "check":
            assert wire["arch"] == "sm_90" and wire["sanitize"] == "memcheck" and "fmt" not in wire
        else:
            assert wire["fmt"] == "ptx" and "arch" not in wire and "sanitize" not in wire
    assert result.facts.disposition == "terminal" and result.transport_status == "succeeded"
    assert all(a.operation == kind and a.idempotency_key == f.bound.idempotency_key
               and a.request_identity == digest(wire) for a in result.attempts)
    assert result.job_references[0].operation == kind
    assert all(c["state"] == "settled" for c in f.state.data["calls"].values())
    fresh = FileExperimentRecords(tmp_path / "records", "cpu", create=False)
    record = fresh.get(f.ref)
    assert record.attempts[0].disposition == "terminal"
    assert SchemaWorkflow._preparation_admission_matches(record.attempts[0], result.job_references[0],
                                                         list(f.state.data["calls"].items()))
    physical = [json.loads(fresh.read_artifact(a)) for a in record.artifacts if a.role == "live_eval_physical"]
    if mode != "accepted":
        assert physical[0]["result"]["transport_status"] == "infrastructure"
        assert physical[0]["result"]["facts"]["disposition"] == "unknown"
        assert not physical[0]["result"]["job_references"]
        assert "503 tengine original evidence" in json.dumps(physical[0])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode", ["overload", "transport", "immediate-400", "persistent", "404"])
def test_sdk_internal_retry_cannot_spend_hidden_posts_or_erase_unknown(tmp_path, monkeypatch, kind, mode):
    f = fixture(tmp_path, monkeypatch, kind, mode=mode)
    result = run(f)
    service = read_service(f)
    count = 3 if mode == "persistent" else 2 if mode == "404" else 1
    assert len([c for c in service["calls"] if c["method"] == "POST"]) == len(f.launches) == count
    assert f.state.data["counts"]["dispatches"] == count
    assert service.get("executions", 0) == 0
    assert not result.job_references
    if mode in {"persistent", "transport", "404"}:
        assert result.facts.disposition == "unknown" and f.state.pending()
        assert result.transport_status == ("rejected" if mode == "404" else "infrastructure")
    else:
        assert result.facts.disposition == "not_submitted" and not f.state.pending()
    if mode == "immediate-400":
        assert result.transport_status == "rejected"
    assert all(c["path"] == PATHS[kind] for c in service["calls"])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("change", ["body", "source", "namespace", "route", "operation", "quota", "deadline"])
def test_live_replay_rejects_changed_identity_or_original_allowance(tmp_path, monkeypatch, kind, change):
    f = fixture(tmp_path, monkeypatch, kind, limit=1 if change == "quota" else 3)
    original = f.executor._run_sandbox
    def physical(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(f.launches) == 1:
            if change in {"body", "source"}:
                prepared = json.loads(f.bound.submission)
                prepared["request"]["candidate" if change == "source" else "name"] = "changed"
                object.__setattr__(f.bound, "submission", json.dumps(prepared).encode())
            elif change == "namespace":
                f.transport.environment["AGATE_AK"] = "other-ak"
            elif change == "route":
                f.transport.endpoint = "https://changed.invalid"
            elif change == "operation":
                object.__setattr__(f.bound, "operation", "run")
            elif change == "deadline":
                kwargs["continuation"].deadline = 0
        return result
    monkeypatch.setattr(f.executor, "_run_sandbox", physical)
    with pytest.raises(Exception):
        run(f)
    assert f.state.data["counts"]["dispatches"] == len(f.launches) == 1
    assert f.state.pending() and f.records.get(f.ref).attempts[0].disposition == "unknown"
    assert len(read_service(f)["calls"]) == 1


@pytest.mark.parametrize("kind", KINDS)
def test_disabled_live_replay_still_preserves_caller_key_and_original_failure(tmp_path, monkeypatch, kind):
    f = fixture(tmp_path, monkeypatch, kind, enabled=False, mode="accepted-loss")
    result = run(f)
    assert result.facts.disposition == "unknown" and result.transport_status == "infrastructure"
    service = read_service(f)
    assert len(service["calls"]) == 1 and service["executions"] == 1
    assert service["calls"][0]["payload"]["idempotency_key"] == f.bound.idempotency_key
    assert result.attempts[0].idempotency_key == f.bound.idempotency_key


@pytest.mark.parametrize("kind", KINDS)
def test_historical_unknown_without_identity_cannot_gain_key_or_resume(tmp_path, kind):
    command = command_at(tmp_path, kind)
    records, ref, attempt = owner_at(tmp_path, command)
    records.observed(ref, attempt, error=RuntimeError("old unretained admission"))
    with pytest.raises(ExperimentRecordError, match="submission_intent_missing"):
        LegacyExperiment._observed_command(command, records, ref, attempt)
    assert not any(a.role == "submission_intent" for a in records.get(ref).artifacts)
    owner = LegacyExperiment(SimpleNamespace(query_job=lambda *a, **k: pytest.fail("unknown Job")), records=records)
    with pytest.raises(ExperimentRecordError, match="experiment_pending"):
        owner._resume_results(records, ref, (command,), None, None, {})


@pytest.mark.parametrize("kind", KINDS)
def test_known_job_uses_existing_resume_query_without_new_post(tmp_path, monkeypatch, kind):
    from aka.execution.gateway.jobs import result_from_job
    f = fixture(tmp_path, monkeypatch, kind, mode="get-failure")
    result = run(f)
    assert result.facts.disposition == "unknown" and len(result.job_references) == 1
    queries = []
    def query(ref, **kwargs):
        queries.append(ref)
        return result_from_job({"job_id": ref.identifier, "status": "succeeded", "result": {"exit_code": 0}}, ref)
    owner = LegacyExperiment(SimpleNamespace(query_job=query), records=f.records)
    owner._resume_results(f.records, f.ref, (f.command,), SimpleNamespace(), SimpleNamespace(step_budgets=()),
                          {"timeout": 60, "remeasurements": 0})
    assert queries == list(result.job_references)
    assert f.records.get(f.ref).attempts[0].disposition == "terminal"
    assert len([c for c in read_service(f)["calls"] if c["method"] == "POST"]) == 1


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("confirmed", [False, True])
def test_cancelled_auxiliary_remains_on_public_original_job_query_path(tmp_path, monkeypatch, kind, confirmed):
    from aka.execution.gateway.jobs import result_from_job
    f = fixture(tmp_path, monkeypatch, kind, mode="cancel-get" if confirmed else "cancel-get-unconfirmed")
    result = run(f)
    assert result.facts.cancellation_requested and len(result.job_references) == 1
    reference, = result.job_references
    queries = []
    def query(ref, **kwargs):
        queries.append(ref)
        return result_from_job({"job_id": ref.identifier, "status": "cancelled"}, ref)
    with f.state.owned():
        assert bool(f.state.pending()) is not confirmed
        f.state.reconcile(SimpleNamespace(query_job=query))
        assert not f.state.pending()
    assert queries == ([] if confirmed else [reference])
    service = read_service(f)
    assert [c["path"] for c in service["calls"] if c["method"] == "POST"] == [
        PATHS[kind], "/v1/jobs/cpu-original-job/cancel"]


@pytest.mark.parametrize("kind", KINDS)
def test_closed_owner_cannot_recreate_authority_after_restart(tmp_path, monkeypatch, kind):
    f = fixture(tmp_path, monkeypatch, kind, mode="persistent")
    assert run(f).facts.disposition == "unknown"
    before = read_service(f)["calls"]
    fresh = FileExperimentRecords(tmp_path / "records", "cpu", create=False)
    rebuilt = LegacyExperiment._observed_command(f.command, fresh, f.ref, f.attempt)
    assert not hasattr(rebuilt.on_observation, "_live_eval_owner")
    with f.state.owned(), f.executor.dispatch_scope(admit=f.state.admit_execution, observe=f.state.observe_execution):
        with pytest.raises(RecoveryRequired):
            f.executor.run(rebuilt)
    assert read_service(f)["calls"] == before


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("accepted_before_loss", [False, True])
def test_real_loopback_http_counts_posts_and_remote_executions(tmp_path, monkeypatch, kind, accepted_before_loss):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    posts, keys, durable, executions = [], set(), [], []
    f = live_fixture(tmp_path, monkeypatch,
        command_factory=lambda path, **kwargs: command_at(path, kind), child_source='''
from aka.execution.gateway import sandbox
sandbox._find_agate = lambda: None
raise SystemExit(sandbox.main())
''')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, value):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            body = json.loads(raw)
            posts.append((self.path, raw))
            record = f.records.get(f.ref)
            frozen, = [json.loads(f.records.read_artifact(a)) for a in record.artifacts
                       if a.role == "live_eval_submission"]
            durable.append(frozen["request"] == body and frozen["identity"]["idempotency_key"] == body["idempotency_key"])
            if len(posts) > 1 or accepted_before_loss:
                if body["idempotency_key"] not in keys:
                    keys.add(body["idempotency_key"])
                    executions.append("executed")
            if len(posts) == 1:
                return self.respond(503, {"message": "CPU receipt unavailable"})
            self.respond(200, {"job_id": "cpu-original-job", "status": "queued"})

        def do_GET(self):
            assert self.path.startswith("/v1/jobs/cpu-original-job")
            self.respond(200, {"job_id": "cpu-original-job", "status": "succeeded", "result": {"exit_code": 0}})

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        serving = Thread(target=server.serve_forever, daemon=True)
        serving.start()
        endpoint = "http://127.0.0.1:" + str(server.server_port)
        f.transport.endpoint = endpoint
        f.executor.configuration["url"] = endpoint
        f.executor.request = replace(f.executor.request, url=endpoint)
        try:
            result = run(f)
        finally:
            server.shutdown()
            serving.join()
    assert len(posts) == len(f.launches) == f.state.data["counts"]["dispatches"] == 2
    assert posts[0] == posts[1] and posts[0][0] == PATHS[kind]
    assert durable == [True, True] and len(executions) == 1
    assert result.facts.disposition == "terminal" and all(c["state"] == "settled" for c in f.state.data["calls"].values())
