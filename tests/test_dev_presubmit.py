"""Real managed Runtime / Legacy construction; only remote submission is synthetic."""
from contextlib import ExitStack
from dataclasses import replace
import json
import hashlib
import subprocess
import sys

import pytest

from aka.contracts.execution import CommandSpec, ResourceIdentity, UnsupportedExecution
from aka.execution.outcomes import captured_result
from aka.workflow.schema.run_state import BudgetDenied, RecoveryRequired
from test_schema_preparation_dispatch import native_launcher


def runtime_launcher(tmp_path, monkeypatch, mode="success"):
    from aka.execution.processes import spawn_owned_session as real_spawn
    campaign, driver, _, launches = native_launcher(tmp_path, monkeypatch)
    runtime = campaign._supervisor_runtime
    runtime.evaluation_configuration = replace(runtime.evaluation_configuration, timeout=3600)
    from schema_cpu_fixture import BASELINE
    campaign.workspace.mkdir(parents=True, exist_ok=True)
    (campaign.workspace / "kernel.py").write_bytes(BASELINE)
    script = """
import json, subprocess, sys
from aka.execution.gateway import sandbox
sandbox._find_agate = lambda: '/cpu/agate'
sandbox._uses_standard_oss_gateway = lambda *a, **kw: False
def submit(**kw):
    argv = kw['agate']
    assert argv[argv.index('--dev-timeout') + 1] == '1800'
    assert argv[argv.index('--job-timeout') + 1] == '1800'
    return subprocess.CompletedProcess(argv, 0, json.dumps({
        'job_id': 'cpu-dev', 'status': 'succeeded',
        'result': {'exit_code': 0, 'stdout': 'cpu complete', 'stderr': ''}}), '')
sandbox._run_agate_once_unchecked = submit
"""
    if mode == "preparation":
        script += "\ndef fail(): raise ValueError('CPU local preparation failed')\nsandbox._find_agate = fail\n"
    elif mode in {"lost-response", "submitted-loss"}:
        script += "\ndef fail(**kw): raise OSError('CPU response lost after submit boundary')\nsandbox._run_agate_once_unchecked = fail\n"
        if mode == "submitted-loss":
            script += "\ndef submitted(**kw):\n    sandbox.emit_fact({'event': 'submitted', 'identifier': 'cpu-lost-job', 'operation': 'dev'})\n    return fail(**kw)\nsandbox._run_agate_once_unchecked = submitted\n"
    elif mode == "timeout":
        # Exercise the launcher's historical 3600/1800 defect independently of
        # the new host-side validation, using the same real Legacy argv builder.
        script += "\ni = sys.argv.index('--timeout'); sys.argv[i + 1] = '3600'\n"
    elif mode == "historical":
        # Reproduce the retained predecessor: no host validation and no owned
        # local-failure fact. All other Runtime/Experiment/sandbox code is real.
        monkeypatch.setattr("aka.execution.gateway.requests.command_timeout",
            lambda command, *a, **kw: command.timeout)
        script += "\nemit = sandbox.emit_fact\nsandbox.emit_fact = lambda value: None if value.get('event') == 'launcher_not_submitted' else emit(value)\n"
    script += "\nraise SystemExit(sandbox.main(sys.argv[1:]))\n"
    def spawn(argv, **kwargs):
        launches.append(tuple(argv))
        return real_spawn([sys.executable, "-c", script, *argv[3:]], **kwargs)
    bound = runtime.executor.executor
    bound.hooks = replace(bound.hooks, spawn=spawn)
    return campaign, driver, runtime, launches


DEV = ["--kind", "dev", "--no-sync", "--", "python3", "kernel.py"]


def test_managed_default_dev_uses_declared_limit_without_changing_eval_budget(tmp_path, monkeypatch):
    campaign, driver, runtime, launches = runtime_launcher(tmp_path, monkeypatch)
    try:
        with driver.state.owned(), ExitStack() as stack:
            driver.state.action("implement")
            driver._scopes(stack, campaign)
            response = runtime.execute_trusted(campaign.workspace, DEV)
            assert response["exit_code"] == 0
            call, = driver.state.data["calls"].values()
            assert call["state"] == "settled" and call["owner"]["timeout"] == 1800
            assert launches[0][launches[0].index("--timeout") + 1] == "1800"
            record, = campaign.experiment_records()
            assert record.lifecycle == "terminal" and record.steps[0].timeout == 1800
            assert record.attempts[0].result.facts.disposition == "terminal"
            from aka.evaluation.tool_requests import request_from_argv
            evaluation = request_from_argv(["--kind", "run", "--no-sync"],
                workspace=campaign.workspace, evaluation_input=runtime.evaluation_input,
                configuration=runtime.evaluation_configuration, workspace_provider=runtime.workspace_provider).request
            assert evaluation.submission is not None
            assert runtime.executor.default_command_timeout(evaluation) == 3600
            assert runtime.evaluation_configuration.timeout == 3600
            assert driver.limits.per_job_timeout_seconds == 3600
    finally:
        runtime.close()


def test_explicit_dev_timeout_is_rejected_before_admission_and_record_settles(tmp_path, monkeypatch):
    campaign, driver, runtime, launches = runtime_launcher(tmp_path, monkeypatch)
    try:
        with driver.state.owned(), ExitStack() as stack:
            driver.state.action("implement")
            driver._scopes(stack, campaign)
            with pytest.raises(UnsupportedExecution, match="3600s exceeds backend limit 1800s"):
                runtime.execute_trusted(campaign.workspace, DEV, execution_timeout=3600)
            assert not launches and driver.state.data["counts"]["dispatches"] == 0
            assert not driver.state.pending()
            record, = campaign.experiment_records()
            assert record.lifecycle == "terminal"
            result = record.attempts[0].result
            assert result.facts.disposition == "not_submitted" and result.facts.local_stopped
            assert result.facts.basis == ("invalid-command-timeout",)
    finally:
        runtime.close()


@pytest.mark.parametrize("mode", ["preparation", "timeout"])
def test_launcher_pre_submit_failure_settles_and_keeps_consumed_admission(tmp_path, monkeypatch, mode):
    campaign, driver, runtime, launches = runtime_launcher(tmp_path, monkeypatch, mode)
    try:
        with driver.state.owned(), ExitStack() as stack:
            driver.state.action("implement")
            driver._scopes(stack, campaign)
            response = runtime.execute_trusted(campaign.workspace, DEV)
            assert response["exit_code"] != 0
            record, = campaign.experiment_records()
            result = record.attempts[0].result
            assert record.lifecycle == "terminal" and record.attempts[0].disposition == "not_submitted"
            assert result.facts.local_stopped and "launcher-before-submission" in result.facts.basis
            assert not result.attempts and not result.job_references
            assert not driver.state.pending() and driver.state.data["counts"]["dispatches"] == 1
            # Root allowance stays consumed; a rejected following Experiment
            # has its own local receipt and cannot become a fake remote pending.
            following = [*DEV[:-1], "-c", "print('different request')"]
            with pytest.raises(BudgetDenied, match="dispatches"):
                runtime.execute_trusted(campaign.workspace, following)
            records = campaign.experiment_records()
            assert len(records) == 2 and all(record.lifecycle == "terminal" for record in records)
            rejected = next(record for record in records if record.ref != records[0].ref)
            assert rejected.attempts[0].result.facts.basis == ("dispatch-not-admitted",)
            assert len(launches) == driver.state.data["counts"]["dispatches"] == 1
    finally:
        runtime.close()


@pytest.mark.parametrize("mode", ["lost-response", "submitted-loss"])
def test_transport_loss_after_submission_boundary_still_blocks_following_work(tmp_path, monkeypatch, mode):
    campaign, driver, runtime, launches = runtime_launcher(tmp_path, monkeypatch, mode)
    try:
        with driver.state.owned(), ExitStack() as stack:
            driver.state.action("implement")
            driver._scopes(stack, campaign)
            runtime.execute_trusted(campaign.workspace, DEV)
            record, = campaign.experiment_records()
            result = record.attempts[0].result
            assert result.facts.disposition == "unknown" and result.facts.local_stopped
            assert bool(result.job_references) == (mode == "submitted-loss")
            assert result.attempts[0].disposition == "unknown"
            assert len(driver.state.pending()) == 1
            following = [*DEV[:-1], "-c", "print('different request')"]
            with pytest.raises(RecoveryRequired, match="Unknown owner"):
                runtime.execute_trusted(campaign.workspace, following)
            assert runtime.execute_trusted(campaign.workspace, following)["exit_code"] != 0
            records = campaign.experiment_records()
            assert len(records) == 2
            rejected = next(item for item in records if item.ref != record.ref)
            assert rejected.lifecycle == "terminal" and rejected.attempts[0].disposition == "not_submitted"
            assert len(rejected.attempts) == 1  # replay of the same failed request cannot loop
            assert len(launches) == driver.state.data["counts"]["dispatches"] == 1
            assert len(driver.state.pending()) == 1
    finally:
        runtime.close()


@pytest.mark.parametrize("extra", [[], [{"event": "submission_started"}],
    [{"event": "submission_intent", "attempt_id": "a"}],
    [{"event": "submitted", "identifier": "cpu-submitted"}], [{"event": "invalid"}],
    [{"event": "submission_started"},
        {"event": "diagnostic", "record": {"failure_class": "request", "reason": "lost-response"}}]])
def test_only_complete_nonconflicting_pre_submit_fact_is_authoritative(extra):
    process = subprocess.CompletedProcess([], 1, "", "local failure")
    records = [{"event": "launcher_not_submitted"}, *extra]
    result = captured_result(process, resource=ResourceIdentity("legacy"), records=records, decode_output=False)
    assert result.facts.disposition == ("not_submitted" if not extra else "unknown")
    running = captured_result(process, resource=ResourceIdentity("legacy"), records=records,
        local_stopped=False, decode_output=False)
    assert running.facts.disposition == "unknown"
    missing = captured_result(process, resource=ResourceIdentity("legacy"), decode_output=False)
    assert missing.facts.disposition == "unknown"


def owner_receipt(experiment, record, *, basis="legacy-dev-timeout-before-submit"):
    records = experiment.records
    raw = (records._directory(record.ref) / "record.json").read_bytes()
    evidence = b'{"cpu_original_owner": "retained process and call-order audit"}'
    proof = {"format": "aka.unsubmitted-owner.v1", "experiment": record.ref.as_dict(),
        "attempt_id": record.attempts[0].attempt_id, "original_record_sha256": hashlib.sha256(raw).hexdigest(),
        "evidence_sha256": hashlib.sha256(evidence).hexdigest(), "basis": basis,
        "reason": "CPU original-owner audit: stopped before launcher submission",
        "original_processes": [{"pid": 2147483647, "start_token": "cpu-original-owner"}],
        "local_process_tree_terminated": True, "submission_boundary_entered": False, "uploads_started": False,
        "call_order": ["local argument construction failed", "transport invocation not entered"],
        "installed_code_sha256": {"sandbox.py": "a" * 64}}
    if record.attempts[0].result:
        proof["stderr_sha256"] = hashlib.sha256(record.attempts[0].result.stderr.encode()).hexdigest()
    return proof, evidence, raw


def test_explicit_owner_and_schema_settlement_preserve_original_result_and_consumption(tmp_path, monkeypatch):
    campaign, driver, runtime, launches = runtime_launcher(tmp_path, monkeypatch, "historical")
    try:
        with driver.state.owned(), ExitStack() as stack:
            driver.state.action("implement")
            driver._scopes(stack, campaign)
            runtime.execute_trusted(campaign.workspace, DEV, execution_timeout=3600)
            permit, = driver.state.pending()
        record, = campaign.experiment_records()
        assert record.attempts[0].disposition == "unknown"
        counts = dict(driver.state.data["counts"])
        source = (campaign.workspace / "kernel.py").read_bytes()
        with runtime.measurements._open() as experiment:
            proof, evidence, raw = owner_receipt(experiment, record)
            receipt = json.dumps(proof).encode()
            # Identity, audit and actual local traceback all matter; missing
            # references alone and a caller-supplied digest do not suffice.
            for key, value in (("submission_boundary_entered", True), ("uploads_started", True),
                    ("local_process_tree_terminated", False), ("original_record_sha256", "b" * 64),
                    ("stderr_sha256", "c" * 64)):
                with pytest.raises(ValueError):
                    experiment.settle_unsubmitted(record.ref, record.attempts[0].attempt_id,
                        receipt=json.dumps({**proof, key: value}).encode(), evidence=evidence)
                assert (experiment.records._directory(record.ref) / "record.json").read_bytes() == raw
            settled = experiment.settle_unsubmitted(record.ref, record.attempts[0].attempt_id,
                receipt=receipt, evidence=evidence)
            assert settled.lifecycle == "terminal" and settled.attempts[0].disposition == "not_submitted"
            original = next(item for item in settled.artifacts if item.role == "unsubmitted-original-record")
            assert experiment.read_artifact(original) == raw
            assert experiment.command_result(record.ref).facts.disposition == "not_submitted"
            assert experiment.settle_unsubmitted(record.ref, record.attempts[0].attempt_id,
                receipt=receipt, evidence=evidence) == settled
        with pytest.raises(RecoveryRequired, match="identities differ"):
            driver.settle_unsubmitted_execution("missing-permit", campaign=campaign,
                experiment_ref=record.ref, attempt_id=record.attempts[0].attempt_id, reason="CPU owner proof")
        for _ in range(2):
            driver.settle_unsubmitted_execution(permit, campaign=campaign,
                experiment_ref=record.ref, attempt_id=record.attempts[0].attempt_id, reason="CPU owner proof")
        assert not driver.state.pending() and driver.state.data["counts"] == counts
        assert len(driver.state.data["unsubmitted_execution_settlements"]) == 1
        assert (campaign.workspace / "kernel.py").read_bytes() == source and len(launches) == 1
    finally:
        runtime.close()


def test_old_admission_rejections_require_owner_proof_and_settle_each_original_attempt(tmp_path):
    from aka.experiment.file_records import FileExperimentRecords
    from aka.experiment.runner import LegacyExperiment
    class Reject:
        def run(self, command):
            raise RecoveryRequired("Unknown owner state prevents another dispatch")
    records = FileExperimentRecords(tmp_path / "records", str(tmp_path))
    experiment = LegacyExperiment(Reject(), records=records)
    for ordinal in range(2):
        command = CommandSpec(tmp_path, ("true", str(ordinal)), operation="dev", timeout=3600, hardware="L20D")
        with pytest.raises(RecoveryRequired):
            experiment.run_recorded(command, request_id=f"rejected-{ordinal}")
        original = experiment.find(f"rejected-{ordinal}")
        assert original.attempts[0].result is None and original.lifecycle == "failed"
        proof, evidence, raw = owner_receipt(experiment, original, basis="legacy-admission-rejected")
        final = experiment.settle_unsubmitted(original.ref, original.attempts[0].attempt_id,
            receipt=json.dumps(proof).encode(), evidence=evidence)
        assert final.lifecycle == "terminal" and final.attempts[0].disposition == "not_submitted"
        assert experiment.command_result(final.ref).transport_status == "rejected"
        assert any(experiment.read_artifact(item) == raw for item in final.artifacts
            if item.role == "unsubmitted-original-record")
    assert not experiment.unresolved() and len(tuple(experiment.iter_records())) == 2


def test_owner_settlement_cannot_clear_a_real_post_submit_unknown(tmp_path, monkeypatch):
    campaign, driver, runtime, _ = runtime_launcher(tmp_path, monkeypatch, "lost-response")
    try:
        with driver.state.owned(), ExitStack() as stack:
            driver.state.action("implement")
            driver._scopes(stack, campaign)
            runtime.execute_trusted(campaign.workspace, DEV)
        original, = campaign.experiment_records()
        with runtime.measurements._open() as experiment:
            proof, evidence, raw = owner_receipt(experiment, original)
            with pytest.raises(ValueError, match="unresolved submission"):
                experiment.settle_unsubmitted(original.ref, original.attempts[0].attempt_id,
                    receipt=json.dumps(proof).encode(), evidence=evidence)
            assert (experiment.records._directory(original.ref) / "record.json").read_bytes() == raw
        assert len(driver.state.pending()) == 1
    finally:
        runtime.close()
