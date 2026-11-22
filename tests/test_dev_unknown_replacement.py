"""CPU native admission: a new request ID cannot clear an unknown Dev effect."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
import json
import multiprocessing
import sys
from threading import Event
from types import SimpleNamespace

import pytest

from aka.contracts.execution import CommandSpec, ExecutionFacts, ExecutionFile, JobResult
from aka.contracts.execution_recovery import JobReference
from aka.contracts.experiment import ExperimentRecordError
from aka.execution.gateway.jobs import result_from_job
from aka.experiment.file_records import FileExperimentRecords
from aka.experiment.runner import LegacyExperiment
from aka.workflow.light.orchestration.supervisor_runtime import Capability, _OperationHandler
from aka.workflow.schema.run_state import RunLimits
from schema_cpu_fixture import BASELINE
from test_schema_preparation_dispatch import native_launcher
from test_schema_workflow import controller


DEV = ["--kind", "dev", "--no-sync", "--", "python3", "kernel.py"]


def native_dev(tmp_path, monkeypatch):
    """Real tool -> Runtime -> Experiment -> budget -> launcher -> fact pipe."""
    from aka.execution.processes import spawn_owned_session as real_spawn
    campaign, original, _, launches = native_launcher(tmp_path, monkeypatch)
    driver, _ = controller(tmp_path, campaign, limits=RunLimits(max_dispatches=8))
    driver.executor = driver._recovery_executor = original.executor
    runtime = campaign._supervisor_runtime
    campaign.workspace.mkdir(parents=True, exist_ok=True)
    (campaign.workspace / "kernel.py").write_bytes(BASELINE)
    executor = runtime.executor.executor
    eval_spawn = executor.hooks.spawn
    script = """
import json, subprocess, sys
from aka.execution.gateway import sandbox
sandbox._find_agate = lambda: '/cpu/agate'
sandbox._uses_standard_oss_gateway = lambda *a, **kw: False
sandbox._submission_route = lambda *a, **kw: ('https://cpu.invalid', '')
def submit(**kw):
    return subprocess.CompletedProcess(kw['agate'], 0, json.dumps({
        'job_id': 'dv_cpu_unknown', 'status': 'failed',
        'execution_state': 'unknown', 'replay_policy': 'never',
        'error': {'category': 'infra', 'code': 'dev_submission_ambiguous',
                  'details': {'retry_advice': 'never'}},
        'cleanup_pending': False}), '')
sandbox._run_agate_once_unchecked = submit
raise SystemExit(sandbox.main(sys.argv[1:]))
"""
    def spawn(argv, **kwargs):
        if argv[argv.index("--kind") + 1] != "dev":
            return eval_spawn(argv, **kwargs)
        launches.append(tuple(argv))
        return real_spawn([sys.executable, "-c", script, *argv[3:]], **kwargs)
    executor.hooks = replace(executor.hooks, spawn=spawn)
    capability = Capability(campaign.workspace.resolve(), {})
    runtime.capabilities["cpu-coder"] = capability
    handler = _OperationHandler(runtime, capability, "/v1/gateway/execute")
    context = SimpleNamespace(workspace=capability.workspace, cancel_event=Event(),
                              filesystem_access=None, deadline=None)
    def tool(argv):
        return json.loads(handler.handle(json.dumps({"argv": argv}).encode(), context))
    return campaign, driver, runtime, launches, tool


def test_real_tool_rejects_new_id_after_terminal_unknown_without_charge_or_global_freeze(tmp_path, monkeypatch):
    campaign, driver, runtime, launches, tool = native_dev(tmp_path, monkeypatch)
    try:
        with driver.state.owned(), ExitStack() as stack:
            driver.state.action("implement")
            driver._scopes(stack, campaign)
            tool(DEV)
            original, = campaign.experiment_records()
            result = original.attempts[0].result
            assert original.lifecycle == "terminal" and result.facts.disposition == "terminal"
            assert result.facts.remote_terminated is None
            assert result.transport_status == "infrastructure"
            from aka.experiment.retry import replay_allowed
            assert not replay_allowed(result.as_dict())
            with runtime.measurements._open() as experiment:
                record_path = experiment.records._directory(original.ref) / "record.json"
                frozen = record_path.read_bytes()
            counts = dict(driver.state.data["counts"])
            assert counts["dispatches"] == 1 and not driver.state.pending()

            # The real Agent handler chooses a fresh non-trusted request ID.
            blocked = tool(DEV)
            assert blocked["status"] == 400
            error = blocked["result"]["error"]
            assert error["code"] == "dev_replacement_unresolved"
            assert error["gateway_record_id"] == "gateway-" + original.ref.experiment_id
            assert "independent measurement" in error["next_action"]
            assert "pivot" in error["next_action"]
            assert driver.state.data["counts"] == counts and not driver.state.pending()
            assert len(launches) == len(campaign.experiment_records()) == 1
            assert record_path.read_bytes() == frozen

            # The original unknown must not freeze this independent full Eval.
            following = tool(["--kind", "run", "--mode", "full", "--no-sync"])
            assert following["status"] == 200 and following["execution_exit_code"] == 0
            assert len(launches) == driver.state.data["counts"]["dispatches"] == 2
            assert driver.state.data["phase"] != "recovery" and not driver.state.pending()
            assert record_path.read_bytes() == frozen
    finally:
        runtime.close()


def command_at(path):
    return CommandSpec(path, ("python3", "probe.py"), operation="dev", timeout=60,
        input_files=(ExecutionFile("probe.py", b"print('CPU fixture')\n"),))


def receipt(*, state="unknown", status="failed", code=None, remote=None):
    job = {"job_id": "dv_cpu_original", "status": status,
           "execution_state": state, "replay_policy": "never", "cleanup_pending": False}
    if code is not None:
        job["result"] = {"exit_code": code}
    if status == "failed":
        job["error"] = {"category": "infra", "code": "dev_submission_ambiguous" if state == "unknown" else "cpu_failure"}
    if remote is not None:
        job["remote_terminated"] = remote
    return result_from_job(job, JobReference("gateway", "dv_cpu_original", operation="dev"), operation="dev")


class CpuExecutor:
    def __init__(self, result):
        self.result, self.dispatched = result, 0

    def run(self, command):
        self.dispatched += 1
        return self.result


def attempt(experiment, command, request_id):
    from aka.contracts.execution import ExecutionUnavailable
    try:
        return experiment.run_recorded(command, request_id=request_id)
    except ExecutionUnavailable:
        return experiment.find(request_id)


@pytest.mark.parametrize("original", [
    receipt(state="stopped", status="succeeded", code=0),
    receipt(state="stopped", status="succeeded"),
    receipt(state="stopped", status="succeeded", code=1),
    receipt(state="stopped", remote=True),
    receipt(remote=True),
    JobResult(1, transport_status="rejected", facts=ExecutionFacts("not_submitted", local_stopped=True)),
])
def test_stopped_and_unsubmitted_dev_allow_fresh_attempts_despite_never(tmp_path, original):
    records = FileExperimentRecords(tmp_path / "records", "cpu")
    executor = CpuExecutor(original)
    experiment = LegacyExperiment(executor, records=records)
    first = attempt(experiment, command_at(tmp_path), "first")
    raw = (records._directory(first.ref) / "record.json").read_bytes()
    second = attempt(experiment, command_at(tmp_path), "second")
    assert first.ref != second.ref and executor.dispatched == 2
    assert first.request_digest == second.request_digest
    assert (records._directory(first.ref) / "record.json").read_bytes() == raw


def test_reopened_native_records_block_new_id_but_keep_original_read_only(tmp_path):
    command = command_at(tmp_path)
    records = FileExperimentRecords(tmp_path / "records", "cpu")
    first = attempt(LegacyExperiment(CpuExecutor(receipt()), records=records), command, "first")
    path = records._directory(first.ref) / "record.json"
    raw = path.read_bytes()
    reopened = FileExperimentRecords(tmp_path / "records", "cpu", create=False)
    executor = CpuExecutor(receipt(status="succeeded", state="stopped", code=0))
    experiment = LegacyExperiment(executor, records=reopened)
    with pytest.raises(ExperimentRecordError, match="dev_replacement_unresolved") as blocked:
        experiment.run_recorded(command, request_id="new-id-after-restart")
    assert blocked.value.ref == first.ref and executor.dispatched == 0
    assert len(tuple(reopened.iter_records())) == 1 and path.read_bytes() == raw
    # Reading/recovering the original cannot create a fresh physical attempt.
    with pytest.raises(ExperimentRecordError, match="dev_replacement_unresolved"):
        experiment.resume_recorded(command, request_id="first", max_remeasurements=3)
    assert executor.dispatched == 0 and path.read_bytes() == raw


@pytest.mark.parametrize("signal", ["policy", "cause"])
def test_either_retained_unknown_signal_blocks_and_changed_source_can_proceed(tmp_path, signal):
    original = receipt()
    original = replace(original, attempts=(), **({"causes": ()} if signal == "policy" else {"diagnostics": ()}))
    records = FileExperimentRecords(tmp_path / "records", "cpu")
    executor = CpuExecutor(original)
    experiment = LegacyExperiment(executor, records=records)
    command = command_at(tmp_path)
    first = attempt(experiment, command, "first")
    with pytest.raises(ExperimentRecordError, match="dev_replacement_unresolved"):
        experiment.run_recorded(command, request_id="same-body")
    executor.result = receipt(state="stopped", status="succeeded", code=0)
    changed = replace(command, input_files=(ExecutionFile("probe.py", b"print('different CPU probe')\n"),))
    second = experiment.run_recorded(changed, request_id="changed-source")
    assert first.request_digest != second.request_digest and executor.dispatched == 2
    assert records.get(first.ref).attempts == first.attempts


@pytest.mark.parametrize("has_intent", [False, True])
def test_prepared_without_attempt_is_not_submission_but_intent_remains_unresolved(tmp_path, has_intent):
    records = FileExperimentRecords(tmp_path / "records", "cpu")
    command = command_at(tmp_path)
    executor = CpuExecutor(receipt(status="succeeded", state="stopped", code=0))
    experiment = LegacyExperiment(executor, records=records)
    with records.operation("old") as ref:
        records.register(ref, "old", experiment._request_data(command), kind="command", purpose="development")
        records.prepared(ref, (command,))
        if has_intent:
            records.intent(ref, 0)
    if has_intent:
        with pytest.raises(ExperimentRecordError, match="dev_replacement_unresolved"):
            experiment.run_recorded(command, request_id="new")
        assert executor.dispatched == 0
    else:
        experiment.run_recorded(command, request_id="new")
        assert executor.dispatched == 1


def test_resume_prepared_request_cannot_dispatch_over_later_unknown(tmp_path):
    records = FileExperimentRecords(tmp_path / "records", "cpu")
    command = command_at(tmp_path)
    executor = CpuExecutor(receipt())
    experiment = LegacyExperiment(executor, records=records)
    with records.operation("prepared-before-crash") as ref:
        records.register(ref, "prepared-before-crash", experiment._request_data(command),
                         kind="command", purpose="development")
        records.prepared(ref, (command,))
    original = attempt(experiment, command, "later-unknown")
    raw = (records._directory(original.ref) / "record.json").read_bytes()
    with pytest.raises(ExperimentRecordError, match="dev_replacement_unresolved"):
        experiment.resume_recorded(command, request_id="prepared-before-crash")
    assert executor.dispatched == 1 and records.get(ref).attempts == ()
    assert (records._directory(original.ref) / "record.json").read_bytes() == raw


@pytest.mark.parametrize("state", ["stopped", "unknown"])
def test_original_job_query_stop_clears_retained_unknown_history_but_success_alone_does_not(tmp_path, state):
    class Queried(CpuExecutor):
        def query_job(self, reference, *, timeout):
            assert reference.identifier == "dv_cpu_original"
            return receipt(status="succeeded", state=state)
    command = command_at(tmp_path)
    records = FileExperimentRecords(tmp_path / "records", "cpu")
    executor = Queried(receipt(status="running"))
    experiment = LegacyExperiment(executor, records=records)
    first = attempt(experiment, command, "original")
    assert first.attempts[0].disposition == "unknown"
    recovered = experiment.resume_recorded(command, request_id="original")
    assert recovered.lifecycle == "terminal" and recovered.phase == "complete"
    current = recovered.attempts[0].result
    assert current.facts.remote_terminated is None  # Stop is explicit SDK policy, not an invented exit code.
    assert current.attempts[:len(first.attempts[0].result.attempts)] == first.attempts[0].result.attempts
    assert executor.dispatched == 1
    if state == "unknown":
        with pytest.raises(ExperimentRecordError, match="dev_replacement_unresolved"):
            experiment.run_recorded(command, request_id="replacement")
        assert executor.dispatched == 1
    else:
        executor.result = receipt(status="succeeded", state="stopped", code=0)
        experiment.run_recorded(command, request_id="replacement")
        assert executor.dispatched == 2


def test_different_job_stop_cannot_clear_original_unknown(tmp_path):
    class WrongQuery(CpuExecutor):
        def query_job(self, reference, *, timeout):
            return replace(receipt(status="succeeded", state="stopped", code=0),
                job_references=(JobReference("gateway", "dv_cpu_other", operation="dev"),))
    records = FileExperimentRecords(tmp_path / "records", "cpu")
    command = command_at(tmp_path)
    executor = WrongQuery(receipt(status="running"))
    experiment = LegacyExperiment(executor, records=records)
    attempt(experiment, command, "original")
    with pytest.raises(ExperimentRecordError, match="job_identity_mismatch"):
        experiment.resume_recorded(command, request_id="original")
    with pytest.raises(ExperimentRecordError, match="dev_replacement_unresolved"):
        experiment.run_recorded(command, request_id="replacement")
    assert executor.dispatched == 1


def _hold_request(root, workspace, ready, release):
    records = FileExperimentRecords(root, "cpu")
    class Held(CpuExecutor):
        def run(self, command):
            ready.set()
            assert release.wait(10)
            return super().run(command)
    attempt(LegacyExperiment(Held(receipt()), records=records), command_at(workspace), "owner")


def test_cross_process_inflight_lock_fails_promptly_then_retained_unknown_blocks(tmp_path):
    command = command_at(tmp_path)
    process_context = multiprocessing.get_context("spawn")
    ready, release = process_context.Event(), process_context.Event()
    process = process_context.Process(target=_hold_request,
        args=(tmp_path / "records", tmp_path, ready, release))
    process.start()
    try:
        assert ready.wait(10)
        records = FileExperimentRecords(tmp_path / "records", "cpu")
        executor = CpuExecutor(receipt())
        experiment = LegacyExperiment(executor, records=records)
        # A timeout makes an accidental blocking flock a bounded test failure.
        with ThreadPoolExecutor() as pool:
            pending = pool.submit(experiment.run_recorded, command, request_id="concurrent")
            try:
                with pytest.raises(ExperimentRecordError, match="dev_request_busy"):
                    pending.result(timeout=2)
            finally:
                release.set()
        process.join(10)
        assert process.exitcode == 0 and executor.dispatched == 0
        with pytest.raises(ExperimentRecordError, match="dev_replacement_unresolved"):
            experiment.run_recorded(command, request_id="after-owner-exit")
        assert len(tuple(records.iter_records())) == 1 and executor.dispatched == 0
    finally:
        release.set()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
