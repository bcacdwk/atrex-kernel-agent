"""Public Schema factory/run/repair with real native preparation and CPU SDK I/O."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import socket
import sys
from threading import Event
from types import SimpleNamespace

import pytest

from aka.bootstrap.runtime import selection_runtime
from aka.bootstrap.continuation import capture
from aka.contracts.startup import Invocation
from aka.contracts.workflow_launch import decode_launch, encode_launch
from aka.core.boot import boot
from aka.execution.legacy.executor import LegacyExecutor
from aka.execution.gateway.jobs import GatewayJobs
from aka.execution.processes import spawn_owned_session
from aka.workflow.schema.launch import prepare_selection
from aka.workflow.schema.run_state import RecoveryRequired
from schema_public_fixture import install_native_cpu, launch, native_selection, schema_config
from test_schema_preparation_dispatch import native_launcher


CHILD = r'''
import contextlib, hashlib, io, json, os, subprocess, sys
from pathlib import Path
def audit(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise AssertionError("CPU fixture forbids network")
sys.addaudithook(audit)
from atrex_gateway_client import Client, cli
from atrex_gateway_client.auth import AkSkAuth
from atrex_gateway_client.errors import GatewayError
from aka.execution.gateway import sandbox
def request(self, method, path, data, *args, **kwargs):
    state = json.loads(state_path.read_text())
    payload = json.loads(data) if data else None
    state["calls"].append({"method": method, "path": path,
        "key": payload.get("idempotency_key") if payload else None,
        "body_sha256": hashlib.sha256(data).hexdigest() if data else None})
    state_path.write_text(json.dumps(state))
    if method == "POST":
        assert path == "/v1/jobs/eval"
        if state["mode"] == "rejected":
            raise GatewayError(400, {"error_class": "validation", "reason": "invalid_source",
                "message": "synthetic worker environment rejected"})
        if state["mode"] == "unknown":
            # No JobRef/key association can turn this physical unknown into a rejection.
            raise GatewayError(503, {"message": "synthetic 503", "execution_state": "unknown"})
        if state["mode"] == "accepted-loss":
            state.setdefault("jobs", {}).setdefault(payload["idempotency_key"], "cpu-prepared-v0")
            state_path.write_text(json.dumps(state))
            if len([c for c in state["calls"] if c["method"] == "POST"]) == 1:
                raise GatewayError(503, {"message": "synthetic original accepted response lost"})
        return 200, {"job_id": "cpu-prepared-v0", "kind": "eval", "status": "queued"}
    assert method == "GET" and path.startswith("/v1/jobs/cpu-prepared-v0")
    if state["mode"] == "get-failure":
        raise GatewayError(401, {"message": "CPU original GET unavailable"})
    return 200, {"job_id": "cpu-prepared-v0", "kind": "eval", "status": "succeeded", "result": {
        "passed": {"compile": {"status": "passed"},
            "correctness": {sid: {"status": "passed"} for sid in ("0", "1", "2")}},
        "performance": {"shapes": {sid: {"samples": [{"end_to_end_time_ms": .01}]}
            for sid in ("0", "1", "2")}}}}
Client._request_once = request
sandbox._find_agate = lambda: "/cpu/agate"
cli._client = lambda args: Client(args.url, auth=AkSkAuth("cpu-ak", "cpu-sk"),
    timeout=1, transport_retries=0)
cli._update.maybe_warn_outdated = lambda version: None
def cpu_cli(argv, **kwargs):
    assert argv[0] == "/cpu/agate"
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            cli.main(argv[1:])
        except SystemExit as stopped:
            code = stopped.code
    return subprocess.CompletedProcess(argv, code, out.getvalue(), err.getvalue())
sandbox.subprocess.run = cpu_cli
raise SystemExit(sandbox.main())
'''


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("prepared recovery CPU tests must not make network calls")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


@contextmanager
def prepared_root(tmp_path, monkeypatch, *, replay=False, mode="rejected", probe=True):
    # Keep the real public composition and driver factory, plus the original
    # native prepare/Experiment/launcher/SDK owners. Substitute endpoint I/O only.
    import atrex_gateway_client
    harness = install_native_cpu(monkeypatch, tmp_path / "public")
    selection, invocation = prepare_selection(native_selection(harness, paired=True),
        Invocation(encode_launch(launch(harness))), schema_config(harness, paired=True))
    selection, _ = capture(selection)
    environment = {"AGATE_AK": "cpu-ak", "AGATE_SK": "cpu-sk", "AGATE_URL": "https://cpu.invalid"}
    native_dir = tmp_path / "native-fixture"
    native_dir.mkdir()
    campaign, _, _, _ = native_launcher(native_dir, monkeypatch, host_environment=environment)
    campaign._supervisor_runtime.close()
    native_measure = campaign.measure_for_acceptance
    probes = []
    def measure(workspace, argv, **kwargs):
        if probe and not probes:
            # A separately requested preparation probe precedes full V0. Its
            # rejected trusted request remains immutable/non-reusable forever.
            argv = list(argv)
            argv[argv.index("--mode") + 1] = "correctness_only"
            probes.append(True)
        return native_measure(workspace, argv, **kwargs)
    campaign.measure_for_acceptance = measure
    factory_owner = campaign.tool_runtime.workflow
    state_path = tmp_path / "cpu-service.json"
    state_path.write_text(json.dumps({"mode": mode, "calls": []}))
    script = tmp_path / "cpu-launcher.py"
    imports = [str(Path(__file__).resolve().parents[1] / "src"),
        str(Path(atrex_gateway_client.__file__).resolve().parents[1])]
    script.write_text("import sys\nsys.path[:0] = " + repr(imports)
        + "\nfrom pathlib import Path\nstate_path = Path(" + repr(str(state_path)) + ")\n" + CHILD)
    launches = []
    def spawn(argv, **kwargs):
        launches.append(tuple(argv))
        return spawn_owned_session([sys.executable, "-B", str(script), *argv[argv.index("--kind"):]], **kwargs)
    monkeypatch.setattr("aka.execution.processes.spawn_owned_session", spawn)
    monkeypatch.setattr(LegacyExecutor, "_job_transport", lambda self:
        SimpleNamespace(environment=environment, endpoint="https://cpu.invalid", profile="",
            idempotent_replay=replay))
    with selection_runtime(selection.environment, selection.python_paths):
        report = boot(selection.composition, tokens=selection.tokens, required_services=selection.required_services)
        try:
            report.verify_composition()
            workflow = report.service("workflow")
            workflow.executor.configuration.update(url="https://cpu.invalid", idempotent_replay=replay)
            campaign.agent = factory_owner.agent = workflow.agent
            factory_owner.executor = workflow.executor
            campaign._supervisor_runtime = None
            campaign.start_runtime()
            request = replace(decode_launch(invocation.argv).request, workspace=campaign.workspace)
            stop = Event()
            stop.set()
            fixture = None
            def reopen():
                # The public restart borrows fresh bindings; references from
                # the previous closed dispatch scope must never be revived.
                campaign._supervisor_runtime.close()
                campaign._supervisor_runtime = None
                campaign.start_runtime()
                if fixture is not None:
                    fixture.runtime = campaign._supervisor_runtime
                driver = workflow.driver_factory(request)
                driver.stop_event = stop
                return driver
            driver = reopen()
            fixture = SimpleNamespace(campaign=campaign, workflow=workflow, driver=driver, reopen=reopen,
                stop=stop, harness=harness, launches=launches, state_path=state_path,
                runtime=campaign._supervisor_runtime)
            yield fixture
        finally:
            campaign._supervisor_runtime.close()
            report.dispose()


def archive(f):
    # Check every original record and artifact byte, not a mutable projection.
    with f.runtime._record_reader() as experiment:
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for record in experiment.iter_records()
            for path in (experiment.records._directory(record.ref) / "record.json",
                *(experiment.records._directory(record.ref) / "artifacts" / item.artifact_id
                  for item in record.artifacts))}


def reject_then_prepare(f):
    with pytest.raises(RuntimeError):
        f.driver.run(f.campaign)
    rejected, = f.campaign.experiment_records()
    assert rejected.attempts[0].disposition == "not_submitted"
    assert not rejected.attempts[0].result.job_references
    call, = f.driver.state.data["calls"].values()
    assert call["state"] == "settled" and call["disposition"] == "not_submitted"
    assert call["recorded_eval"]["attempt_id"] == rejected.attempts[0].attempt_id
    assert len(call["execution_attempts"]) == 1
    assert f.driver.state.data["counts"]["dispatches"] == len(f.launches) == 1
    before = archive(f)
    old_call = deepcopy(call)
    state = json.loads(f.state_path.read_bytes())
    state["mode"] = "accepted"
    f.state_path.write_text(json.dumps(state))
    f.driver = f.reopen()
    f.driver.resume_after_repair("CPU endpoint repaired; retain the settled rejected dispatch")
    assert f.driver.run(f.campaign) == "interrupted: requested"
    assert f.driver.state.data["counts"]["dispatches"] == len(f.launches) == 2
    assert list(f.driver.state.data["calls"].values())[0] == old_call
    assert all(archive(f)[key] == value for key, value in before.items())
    assert len(f.driver.state.data["preparation_measurements"]) == 2
    return rejected


@pytest.mark.parametrize("replay", [False, True], ids=["default-recorded", "live-eval"])
def test_public_prepared_resume_keeps_rejected_dispatch_and_v0(tmp_path, monkeypatch, replay):
    with prepared_root(tmp_path, monkeypatch, replay=replay) as f:
        reject_then_prepare(f)
        before = archive(f)
        retained = deepcopy(f.driver.state.data)
        for _ in range(2):
            f.driver = f.reopen()
            assert f.driver.run(f.campaign) == "interrupted: requested"
            f.driver.resume_after_repair("CPU original owners settled", preparation_campaign=f.campaign)
            data = f.driver.state.data
            assert data["counts"] == retained["counts"] and data["calls"] == retained["calls"]
            assert data["prepared"] == retained["prepared"]
            assert data["preparation_measurements"] == retained["preparation_measurements"]
            assert data["config"] == retained["config"] and data["root_ref"] == retained["root_ref"]
            assert not data.get("preparation_dispatch_corrections")
            assert archive(f) == before and len(f.launches) == 2
        # The real Planner/Agent process stub is reached after the prepared gate.
        f.driver.stop_event = SimpleNamespace(is_set=lambda: bool(f.harness.calls))
        assert f.driver.run(f.campaign) == "interrupted: requested"
        assert [row["kind"] for row in f.harness.calls] == ["optimize"]
        assert f.driver.state.data["counts"]["agent_calls"] == 1
        assert f.driver.state.data["counts"]["dispatches"] == len(f.launches) == 2
        assert archive(f) == before


@pytest.mark.parametrize("broken", [
    "missing-permit", "scope", "native-attempt", "physical-attempt", "key", "body",
    "missing-identity", "duplicate-permit", "unknown-call", "unknown-physical", "not-stopped",
    "success-no-job", "success-unknown", "success-wrong-job", "success-unsettled-permit",
])
def test_prepared_recovery_rejects_unowned_or_conflicting_evidence(tmp_path, monkeypatch, broken):
    with prepared_root(tmp_path, monkeypatch) as f:
        rejected = reject_then_prepare(f)
        before = archive(f)
        records = list(f.campaign.experiment_records())
        index = next(i for i, record in enumerate(records) if record.ref == rejected.ref)
        record = records[index]
        attempt, = record.attempts
        result = attempt.result
        physical, = result.attempts
        # Adversarial projections stay in the CPU reader/in-memory loaded state;
        # original records, root config, Task and frozen requests stay untouched.
        original_reconcile = f.driver.state.reconcile
        def reconcile(*args, **kwargs):
            original_reconcile(*args, **kwargs)
            data = f.driver.state.data
            permit = next(key for key, call in data["calls"].items() if call["disposition"] == "not_submitted")
            call = data["calls"][permit]
            if broken == "missing-permit":
                del data["calls"][permit]
            elif broken == "scope":
                call["owner"]["command_digest"] = "f" * 64
            elif broken == "missing-identity":
                call.pop("recorded_eval")
                call.pop("execution_attempts")
            elif broken == "duplicate-permit":
                data["calls"]["duplicate"] = deepcopy(call)
            elif broken == "unknown-call":
                call["state"] = "unknown"
            elif broken == "success-unsettled-permit":
                next(c for c in data["calls"].values() if c["disposition"] == "terminal")["state"] = "unknown"
        monkeypatch.setattr(f.driver.state, "reconcile", reconcile)
        if broken == "native-attempt":
            attempt = replace(attempt, attempt_id="different-native-attempt")
        elif broken in {"physical-attempt", "key", "body", "unknown-physical"}:
            changes = {"physical-attempt": {"attempt_id": "different-physical-attempt"},
                "key": {"idempotency_key": "another-key"}, "body": {"request_identity": "f" * 64},
                "unknown-physical": {"disposition": "unknown"}}
            attempt = replace(attempt, result=replace(result, attempts=(replace(physical, **changes[broken]),)))
        elif broken == "not-stopped":
            attempt = replace(attempt, result=replace(result, facts=replace(result.facts, local_stopped=False)))
        if broken.startswith("success-"):
            index = 1 - index
            record = records[index]
            attempt, = record.attempts
            result = attempt.result
            if broken == "success-no-job":
                attempt = replace(attempt, result=replace(result, job_references=()))
            elif broken == "success-unknown":
                attempt = replace(attempt, result=replace(result, facts=replace(result.facts, disposition="unknown")))
            elif broken == "success-wrong-job":
                physical, = result.attempts
                attempt = replace(attempt, result=replace(result, attempts=(replace(physical,
                    resource=replace(physical.resource, identifier="another-job")),)))
        records[index] = replace(record, attempts=(attempt,))
        monkeypatch.setattr(f.runtime, "read_experiment_records", lambda: tuple(records))
        with pytest.raises(RecoveryRequired):
            f.driver.run(f.campaign)
        assert f.driver.state.data["counts"]["dispatches"] == len(f.launches) == 2
        assert not f.driver.state.data.get("preparation_dispatch_corrections")
        assert archive(f) == before and not f.harness.calls


def test_unknown_503_without_job_still_requires_original_owner(tmp_path, monkeypatch):
    with prepared_root(tmp_path, monkeypatch, mode="unknown") as f:
        with pytest.raises(RuntimeError):
            f.driver.run(f.campaign)
        record, = f.campaign.experiment_records()
        assert record.attempts[0].disposition == "unknown"
        assert not record.attempts[0].result.job_references
        before = archive(f)
        # A public-looking not_submitted projection cannot settle the actual
        # retained 503 physical unknown or its original unknown permit.
        attempt, = record.attempts
        physical, = attempt.result.attempts
        projected = replace(record, lifecycle="terminal", phase="complete", attempts=(replace(attempt,
            disposition="not_submitted", result=replace(attempt.result,
                facts=replace(attempt.result.facts, disposition="not_submitted"),
                attempts=(replace(physical, idempotency_key=None, request_identity=None),))),))
        for _ in range(2):
            f.driver = f.reopen()
            monkeypatch.setattr(f.runtime, "read_experiment_records", lambda: (projected,))
            with pytest.raises(RecoveryRequired):
                f.driver.resume_after_repair("CPU 503 has no terminal original-owner evidence")
            with pytest.raises(RecoveryRequired):
                f.driver.run(f.campaign)
        assert f.driver.state.data["counts"]["dispatches"] == len(f.launches) == 1
        assert archive(f) == before and not f.harness.calls


@pytest.mark.parametrize("broken", [None, "attempt_id", "idempotency_key", "request_identity", "scope"])
def test_old_live_rejection_uses_its_original_exact_association(tmp_path, monkeypatch, broken):
    with prepared_root(tmp_path, monkeypatch, replay=True) as f:
        reject_then_prepare(f)
        before = archive(f)
        counts = deepcopy(f.driver.state.data["counts"])
        original_reconcile = f.driver.state.reconcile
        def reconcile(*args, **kwargs):
            original_reconcile(*args, **kwargs)
            for call in f.driver.state.data["calls"].values():
                call.pop("execution_attempts")
                call.pop("recorded_eval")
                if call["disposition"] == "not_submitted":
                    if broken == "scope":
                        call["owner"]["command_digest"] = "f" * 64
                    elif broken:
                        call["live_eval"][broken] = "f" * 64
        monkeypatch.setattr(f.driver.state, "reconcile", reconcile)
        if broken:
            with pytest.raises(RecoveryRequired):
                f.driver.run(f.campaign)
        else:
            assert f.driver.run(f.campaign) == "interrupted: requested"
        assert f.driver.state.data["counts"] == counts and len(f.launches) == 2
        assert archive(f) == before and not f.harness.calls


def test_no_physical_receipt_requires_the_existing_explicit_settlement(tmp_path, monkeypatch):
    from test_dev_presubmit import DEV, owner_receipt, runtime_launcher
    campaign, driver, runtime, launches = runtime_launcher(tmp_path, monkeypatch, "historical")
    try:
        with driver.state.owned(), ExitStack() as stack:
            driver.state.action("prepare")
            driver._scopes(stack, campaign)
            runtime.execute_trusted(campaign.workspace, DEV, execution_timeout=3600)
            permit, = driver.state.pending()
        record, = campaign.experiment_records()
        with runtime.measurements._open() as experiment:
            proof, evidence, _ = owner_receipt(experiment, record)
            settled = experiment.settle_unsubmitted(record.ref, record.attempts[0].attempt_id,
                receipt=json.dumps(proof).encode(), evidence=evidence)
        with driver.state.owned(), pytest.raises(RecoveryRequired):
            driver._preparation_dispatch_gaps((settled,))
        for _ in range(2):
            driver.settle_unsubmitted_execution(permit, campaign=campaign,
                experiment_ref=record.ref, attempt_id=record.attempts[0].attempt_id,
                reason="CPU original owner confirmed no submission")
            with driver.state.owned():
                assert driver._preparation_dispatch_gaps((settled,)) == {}
                assert driver.state.data["counts"]["dispatches"] == len(launches) == 1
        with driver.state.owned():
            # Settled-looking facts alone cannot replace the explicit mapping.
            driver.state.data["unsubmitted_execution_settlements"].clear()
            with pytest.raises(RecoveryRequired):
                driver._preparation_dispatch_gaps((settled,))
    finally:
        runtime.close()


@contextmanager
def query_recovery_root(tmp_path, monkeypatch, *, replay=False):
    with prepared_root(tmp_path, monkeypatch, replay=replay, mode="get-failure", probe=False) as f:
        reads = []
        def http(endpoint, method, path, payload, timeout):
            assert endpoint == "https://cpu.invalid"
            assert method == "GET" and path == "/v1/jobs/cpu-prepared-v0"
            reads.append((method, path))
            mode = json.loads(f.state_path.read_bytes())["mode"]
            return {"job_id": "wrong-job" if mode == "wrong-job" else "cpu-prepared-v0",
                "kind": "eval", "status": "queued" if mode == "get-failure" else "succeeded", "result": {
                    "passed": {"compile": {"status": "passed"},
                        "correctness": {sid: {"status": "passed"} for sid in ("0", "1", "2")}},
                    "performance": {"shapes": {sid: {"samples": [{"end_to_end_time_ms": .01}]}
                        for sid in ("0", "1", "2")}}}}
        gateway = GatewayJobs(endpoint="https://cpu.invalid", http=http,
            environment={"AGATE_AK": "cpu-ak", "AGATE_SK": "cpu-sk", "AGATE_URL": "https://cpu.invalid"})
        monkeypatch.setattr(LegacyExecutor, "query_job", lambda self, reference, **kwargs:
            gateway.query(reference, **kwargs))
        f.reads = reads
        with pytest.raises(RuntimeError):
            f.driver.run(f.campaign)
        record, = f.campaign.experiment_records()
        assert record.attempts[0].disposition == "unknown"
        assert len(record.attempts[0].result.job_references) == 1
        assert f.driver.state.data["counts"]["dispatches"] == len(f.launches) == 1
        f.original_record = record
        yield f


def complete_original_query(f):
    state = json.loads(f.state_path.read_bytes())
    state["mode"] = "accepted"
    f.state_path.write_text(json.dumps(state))
    original_artifacts = {path: value for path, value in archive(f).items() if not path.endswith("record.json")}
    f.driver = f.reopen()
    assert f.driver.run(f.campaign) == "interrupted: requested"
    record, = f.campaign.experiment_records()
    assert record.ref == f.original_record.ref and record.request_digest == f.original_record.request_digest
    assert record.lifecycle == record.attempts[0].disposition == "terminal"
    initial = f.original_record.attempts[0]
    resumed, = record.attempts
    assert resumed.attempt_id == initial.attempt_id
    assert resumed.result.attempts[0] == initial.result.attempts[0]
    assert [item.operation for item in resumed.result.attempts] == ["run", "query"]
    assert [item.disposition for item in resumed.result.attempts] == ["unknown", "terminal"]
    assert resumed.result.job_references == initial.result.job_references
    assert all(archive(f)[key] == value for key, value in original_artifacts.items())
    assert len(f.reads) == 2  # Root settlement and native Experiment recovery each query the original Job.
    assert f.driver.state.data["counts"]["dispatches"] == len(f.launches) == 1
    call, = f.driver.state.data["calls"].values()
    assert call["state"] == "settled" and call["execution_attempts"][0]["disposition"] == "unknown"
    return record


@pytest.mark.parametrize("replay", [False, True], ids=["default-recorded", "live-eval"])
def test_public_prepared_original_job_query_survives_save_and_reopen(tmp_path, monkeypatch, replay):
    with query_recovery_root(tmp_path, monkeypatch, replay=replay) as f:
        complete_original_query(f)
        before, retained = archive(f), deepcopy(f.driver.state.data)
        for _ in range(2):
            f.driver = f.reopen()
            assert f.driver.run(f.campaign) == "interrupted: requested"
            f.driver.resume_after_repair("CPU original Job query settled", preparation_campaign=f.campaign)
            for name in ("calls", "counts", "prepared", "preparation_measurements"):
                assert f.driver.state.data[name] == retained[name]
            assert archive(f) == before and not f.driver.state.data.get("preparation_dispatch_corrections")
        f.driver.stop_event = SimpleNamespace(is_set=lambda: bool(f.harness.calls))
        assert f.driver.run(f.campaign) == "interrupted: requested"
        assert [row["kind"] for row in f.harness.calls] == ["optimize"]
        assert len(f.launches) == f.driver.state.data["counts"]["dispatches"] == 1
        assert len(f.reads) == 2 and archive(f) == before
        calls = json.loads(f.state_path.read_bytes())["calls"]
        assert [item["method"] for item in calls] == ["POST", "GET"]


@pytest.mark.parametrize("mode", ["get-failure", "wrong-job"])
def test_public_original_query_must_confirm_its_own_terminal_job(tmp_path, monkeypatch, mode):
    with query_recovery_root(tmp_path, monkeypatch) as f:
        before = archive(f)
        state = json.loads(f.state_path.read_bytes())
        state["mode"] = mode
        f.state_path.write_text(json.dumps(state))
        f.driver = f.reopen()
        with pytest.raises(RecoveryRequired):
            f.driver.run(f.campaign)
        assert archive(f) == before and len(f.reads) == 1
        assert f.driver.state.data["counts"]["dispatches"] == len(f.launches) == 1
        assert not f.driver.state.data.get("prepared") and not f.harness.calls


@pytest.mark.parametrize("broken", [
    "original-no-job", "original-wrong-job", "query-wrong-job", "query-unknown", "query-key",
    "query-body", "query-backend", "missing-query-basis", "extra-dispatch", "unadmitted", "unsettled",
])
def test_prepared_query_chain_rejects_conflicting_or_unadmitted_evidence(tmp_path, monkeypatch, broken):
    with query_recovery_root(tmp_path, monkeypatch) as f:
        record = complete_original_query(f)
        before = archive(f)
        attempt, = record.attempts
        result = attempt.result
        original, query = result.attempts
        if broken.startswith("original-"):
            original = replace(original, resource=replace(original.resource,
                identifier=None if broken == "original-no-job" else "wrong-job"))
        elif broken == "query-wrong-job":
            query = replace(query, resource=replace(query.resource, identifier="wrong-job"))
        elif broken == "query-unknown":
            query = replace(query, disposition="unknown")
        elif broken == "query-key":
            query = replace(query, idempotency_key="wrong-key")
        elif broken == "query-body":
            query = replace(query, request_identity="f" * 64)
        elif broken == "query-backend":
            query = replace(query, resource=replace(query.resource, backend="another-backend"))
        elif broken == "missing-query-basis":
            result = replace(result, facts=replace(result.facts, basis=()))
        receipts = (original, query)
        if broken == "extra-dispatch":
            receipts = (original, replace(original, attempt_id="another-dispatch"), query)
        projected = replace(record, attempts=(replace(attempt, result=replace(result, attempts=receipts)),))
        f.driver = f.reopen()
        monkeypatch.setattr(f.runtime, "read_experiment_records", lambda: (projected,))
        original_reconcile = f.driver.state.reconcile
        def reconcile(*args, **kwargs):
            original_reconcile(*args, **kwargs)
            if broken == "unadmitted":
                f.driver.state.data["calls"].clear()
            elif broken == "unsettled":
                next(iter(f.driver.state.data["calls"].values()))["state"] = "unknown"
        monkeypatch.setattr(f.driver.state, "reconcile", reconcile)
        with pytest.raises(RecoveryRequired):
            f.driver.run(f.campaign)
        assert archive(f) == before and len(f.reads) == 2 and not f.harness.calls
        assert f.driver.state.data["counts"]["dispatches"] == len(f.launches) == 1
        assert not f.driver.state.data.get("preparation_dispatch_corrections")


@pytest.mark.parametrize("broken", [None, "missing-association", "wrong-key", "wrong-permit-group",
    "missing-permit", "duplicate-permit", "unsettled", "missing-physical-observation", "reordered"])
def test_public_prepared_keyed_live_recovery_preserves_every_paid_post(tmp_path, monkeypatch, broken):
    from aka.execution import live_eval
    from aka.contracts.execution_recovery import InfrastructureRetryPolicy
    monkeypatch.setattr(live_eval, "InfrastructureRetryPolicy", lambda: InfrastructureRetryPolicy((0, 0), 120))
    with prepared_root(tmp_path, monkeypatch, replay=True, mode="accepted-loss", probe=False) as f:
        assert f.driver.run(f.campaign) == "interrupted: requested"
        record, = f.campaign.experiment_records()
        assert record.lifecycle == record.attempts[0].disposition == "terminal"
        assert len(record.attempts[0].result.attempts) == 2
        assert len(record.attempts[0].result.job_references) == 1
        with f.runtime._record_reader() as experiment:
            physical = [json.loads(experiment.read_artifact(item)) for item in record.artifacts
                if item.role == "live_eval_physical"]
        assert len(physical) == 2 and physical[0]["result"]["facts"]["disposition"] == "unknown"
        assert physical[0]["result"]["job_references"] == []
        before, retained = archive(f), deepcopy(f.driver.state.data)
        assert len(retained["calls"]) == retained["counts"]["dispatches"] == len(f.launches) == 2
        assert all(call["state"] == "settled" for call in retained["calls"].values())
        if broken and broken != "reordered":
            f.driver = f.reopen()
            original_reconcile = f.driver.state.reconcile
            def reconcile(*args, **kwargs):
                original_reconcile(*args, **kwargs)
                calls = f.driver.state.data["calls"]
                first, second = calls.values()
                if broken == "missing-association":
                    first.pop("live_eval_association")
                elif broken == "wrong-key":
                    first["live_eval"]["idempotency_key"] = "wrong-key"
                elif broken == "wrong-permit-group":
                    second["original_permit"] = next(reversed(calls))
                elif broken == "missing-permit":
                    calls.pop(next(iter(calls)))
                elif broken == "duplicate-permit":
                    calls["duplicate"] = deepcopy(second)
                elif broken == "unsettled":
                    first["state"] = "unknown"
                elif broken == "missing-physical-observation":
                    first.pop("execution_attempts")
            monkeypatch.setattr(f.driver.state, "reconcile", reconcile)
            with pytest.raises(RecoveryRequired):
                f.driver.run(f.campaign)
            assert archive(f) == before and not f.harness.calls
            assert f.driver.state.data["counts"]["dispatches"] == len(f.launches) == 2
            assert not f.driver.state.data.get("preparation_dispatch_corrections")
            return
        for _ in range(2):
            f.driver = f.reopen()
            if broken == "reordered":
                # JSON object order is not causal order (notably permits 9/10).
                original_reconcile = f.driver.state.reconcile
                def reconcile(*args, **kwargs):
                    original_reconcile(*args, **kwargs)
                    calls = f.driver.state.data["calls"]
                    f.driver.state.data["calls"] = dict(reversed(list(calls.items())))
                monkeypatch.setattr(f.driver.state, "reconcile", reconcile)
            assert f.driver.run(f.campaign) == "interrupted: requested"
            f.driver.resume_after_repair("CPU original live Eval completed", preparation_campaign=f.campaign)
            assert f.driver.state.data["calls"] == retained["calls"]
            assert f.driver.state.data["counts"] == retained["counts"]
            assert archive(f) == before and len(f.launches) == 2
            assert not f.driver.state.data.get("preparation_dispatch_corrections")
        posts = [item for item in json.loads(f.state_path.read_bytes())["calls"] if item["method"] == "POST"]
        assert len(posts) == 2 and posts[0] == posts[1]
        assert len(json.loads(f.state_path.read_bytes())["jobs"]) == 1
