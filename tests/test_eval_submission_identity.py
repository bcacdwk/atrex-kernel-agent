"""CPU receipts for admission loss: retain identity without authorizing a retry."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from aka.contracts.execution import CommandSpec, PreparedSourceArchive, ResourceIdentity
from aka.contracts.experiment import ExperimentRecordError
from aka.execution import outcomes
from aka.execution.gateway import requests, sandbox, submission
from aka.execution.gateway.source_archives import prepared_payload
from aka.experiment.file_records import FileExperimentRecords
from aka.experiment.runner import LegacyExperiment


def command_at(path, *, abba=False):
    request = {"name": "cpu-eval", "candidate": "# candidate", "mode": "full",
        "spec": {"target_hardware": ["L20D"], "num_gpus": 1, "languages": ["triton"]},
        "reference": {"operator": "cpu", "reference_py": "# reference", "input_py": "# inputs",
                      "shapes": {"0": {"N": 32}}},
        "requirements": ["cpu-fixture==1"], "deps_mode": "freeze_installed",
        "options": {"timeout_s": 60, "num_correctness_cases": 6, "bench_iters": 100,
                    "atol": 0.0001, "rtol": 0.001},
        "env_vars": {"TRITON_CPU_FIXTURE": "1"},
        "runner_overrides": {"skip_kernel_attribution": True, "warmup_iters": 7}}
    if abba:
        request["abba"] = {"baseline": "# baseline", "repeats": 2}
    return CommandSpec(path, ("python3", "test_kernel.py"), timeout=60, hardware="L20D",
        operation="same_allocation_abba" if abba else "run",
        submission=json.dumps({"schema_version": 1, "kind": "run", "request": request}).encode())


def owner_at(path, command):
    records = FileExperimentRecords(path / "records", "cpu")
    with records.operation("eval") as ref:
        records.register(ref, "eval", {"cpu": True}, kind="command", purpose="formal")
        records.prepared(ref, (command,))
        attempt = records.intent(ref, 0)
    return records, ref, attempt


def retained_binding(records, ref):
    artifact, = (item for item in records.get(ref).artifacts if item.role == "submission_intent")
    return json.loads(records.read_artifact(artifact)), artifact


def invoke(command):
    args = SimpleNamespace(kind=command.operation, url="https://cpu.invalid", gateway_profile=None,
                           ssh=False, input=[], timeout=command.timeout)
    return submission.submit_prepared(args, command.workspace, 0, json.loads(prepared_payload(command)))


@pytest.mark.parametrize("abba", [False, True])
@pytest.mark.parametrize("failure", ["503", "crash", "accepted"])
def test_sdk_wire_has_durable_key_before_post_even_without_fact_pipe(tmp_path, monkeypatch, abba, failure):
    cli = pytest.importorskip("atrex_gateway_client.cli")
    from atrex_gateway_client import Client
    from atrex_gateway_client.errors import GatewayError

    command = command_at(tmp_path, abba=abba)
    records, ref, attempt = owner_at(tmp_path, command)
    bound = LegacyExperiment._observed_command(command, records, ref, attempt)
    client = Client("https://cpu.invalid", timeout=1, transport_retries=0)
    monkeypatch.setattr(cli, "_client", lambda args: client)
    monkeypatch.setattr(cli._update, "maybe_warn_outdated", lambda version: None)
    monkeypatch.setattr(sandbox, "_find_agate", lambda: "/cpu/agate")
    monkeypatch.setattr(sandbox, "_submission_route", lambda *args: ("https://cpu.invalid", ""))
    facts, wire_bodies = [], []

    def fact(value):
        if failure != "crash":
            facts.append({"attempt_id": outcomes._attempt.get(), **value})

    for module in (outcomes, sandbox, submission, requests):
        monkeypatch.setattr(module, "emit_fact", fact)

    def post(method, path, data, *args, **kwargs):
        assert (method, path) == ("POST", "/v1/jobs/eval")
        # A fresh reader proves publication completed before the very first POST.
        reopened = FileExperimentRecords(tmp_path / "records", "cpu", create=False)
        binding, artifact = retained_binding(reopened, ref)
        assert binding["attempt_id"] == attempt
        assert binding["submission_sha256"] == hashlib.sha256(command.submission).hexdigest()
        assert not (reopened.root / ref.experiment_id / "artifacts" / artifact.artifact_id).stat().st_mode & 0o222
        payload = json.loads(data)
        assert binding["idempotency_key"] == bound.idempotency_key == payload["idempotency_key"]
        original = json.loads(command.submission)["request"]
        assert payload == {**original, "idempotency_key": bound.idempotency_key}
        wire_bodies.append(data)
        if failure == "503":
            raise GatewayError(503, {"message": "<html>503 Service Temporarily Unavailable tengine</html>"})
        if failure == "crash":
            raise KeyboardInterrupt("CPU lost owner and fact pipe after POST")
        return 200, {"job_id": "cpu-original-job", "status": "queued", "kind": "eval"}

    monkeypatch.setattr(client, "_request_once", post)

    def run(argv, **kwargs):
        assert argv[1] == "submit" and argv[-1] == "--no-wait"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr), pytest.raises(SystemExit) as stopped:
            cli.main(argv[1:])
        return subprocess.CompletedProcess(argv, stopped.value.code, stdout.getvalue(), stderr.getvalue())

    monkeypatch.setattr(sandbox.subprocess, "run", run)
    monkeypatch.setattr(sandbox, "_resume_interrupted_agate_wait", lambda **kwargs:
        subprocess.CompletedProcess((), 0, json.dumps({"job_id": "cpu-original-job",
            "status": "succeeded", "result": {}}), ""))
    if failure == "crash":
        with pytest.raises(KeyboardInterrupt):
            invoke(bound)
        assert not facts
        records.observed(ref, attempt, error=RuntimeError("CPU owner stopped"))
        fresh = FileExperimentRecords(tmp_path / "records", "cpu", create=False)
        binding, _ = retained_binding(fresh, ref)
        source, = (item for item in fresh.get(ref).artifacts
                   if item.sha256 == binding["submission_sha256"])
        recovered = replace(command, submission=fresh.read_artifact(source))
        recovered = LegacyExperiment._observed_command(recovered, fresh, ref, attempt)
        payload = json.loads(prepared_payload(recovered))["request"]
        assert json.loads(wire_bodies[0]) == payload
        # Identity recovery does not authorize a second physical submission.
        experiment = LegacyExperiment(SimpleNamespace(), records=fresh)
        with pytest.raises(ExperimentRecordError, match="experiment_pending"):
            experiment._resume_results(fresh, ref, (command,), None, None, {})
    else:
        returncode = invoke(bound)
        receipt = outcomes.captured_result(subprocess.CompletedProcess((), returncode, "", ""),
            resource=ResourceIdentity("gateway"), records=facts, decode_output=False)
        assert receipt.attempts[0].idempotency_key == bound.idempotency_key
        if failure == "503":
            assert receipt.facts.disposition == "unknown"
            assert receipt.transport_status == "infrastructure" and not receipt.job_references
        else:
            reference, = receipt.job_references
            assert reference.operation == "run" and reference.idempotency_key == bound.idempotency_key
            assert reference.identifier == "cpu-original-job"
    assert len(wire_bodies) == 1


@pytest.mark.parametrize("change", ["payload", "timeout", "environment", "key"])
def test_existing_attempt_rejects_changed_binding(tmp_path, change):
    command = command_at(tmp_path)
    records, ref, attempt = owner_at(tmp_path, command)
    bound = LegacyExperiment._observed_command(command, records, ref, attempt)
    changes = {"payload": {"submission": command.submission + b" "}, "timeout": {"timeout": 61},
               "environment": {"environment": {"AGATE_URL": "https://other.invalid"}},
               "key": {"idempotency_key": "another-key"}}
    with pytest.raises(ExperimentRecordError, match="submission_identity_mismatch"):
        LegacyExperiment._observed_command(replace(bound, **changes[change]), records, ref, attempt)


def test_new_intents_have_distinct_keys_but_old_unknown_cannot_gain_one(tmp_path):
    command = command_at(tmp_path)
    records, ref, first = owner_at(tmp_path, command)
    first_bound = LegacyExperiment._observed_command(command, records, ref, first)
    # This is explicit owner admission, not a retry loop or unknown settlement.
    second = records.intent(ref, 0)
    second_bound = LegacyExperiment._observed_command(command, records, ref, second)
    assert first_bound.idempotency_key != second_bound.idempotency_key
    old = records.intent(ref, 0)
    records.observed(ref, old, error=RuntimeError("historical unretained submission"))
    with pytest.raises(ExperimentRecordError, match="submission_intent_missing"):
        LegacyExperiment._observed_command(command, records, ref, old)


def test_hard_limit_change_cannot_change_payload_under_original_key(tmp_path, monkeypatch):
    command = command_at(tmp_path)
    records, ref, attempt = owner_at(tmp_path, command)
    bound = LegacyExperiment._observed_command(command, records, ref, attempt)
    monkeypatch.setenv(requests.LIMITS_ENV, json.dumps({"max_job_timeout_s": 59}))
    monkeypatch.setattr(sandbox, "_run_agate_with_cancel_retry", lambda **kwargs: pytest.fail("unexpected POST"))
    with pytest.raises(ValueError, match="allocation exceeds"):
        invoke(bound)


def test_unrecorded_and_non_inline_commands_do_not_gain_inline_keys(tmp_path):
    command = command_at(tmp_path)
    assert LegacyExperiment._observed_command(command, None, None, None) is command
    records, ref, attempt = owner_at(tmp_path, command)
    dev = replace(command, operation="dev", submission=None)
    assert LegacyExperiment._observed_command(dev, records, ref, attempt).idempotency_key is None
    assert not any(item.role == "submission_intent" for item in records.get(ref).artifacts)


@pytest.mark.parametrize("profile", [False, True])
def test_archive_keeps_original_key_derivation_and_upload_route(tmp_path, monkeypatch, profile):
    from aka.execution.gateway import source_archives
    from aka.contracts.execution import ProfileOptions
    command = command_at(tmp_path)
    prepared = json.loads(command.submission)
    if profile:
        prepared["kind"] = "profile"
        command = replace(command, operation="profile", profile_options=ProfileOptions("deep", "ncu", ("cpu.counter",)))
    prepared["request"]["candidate"] = {"archive": "source.tar.gz", "entry_point": "kernel.py"}
    archive = PreparedSourceArchive("candidate", "source.tar.gz", "kernel.py", b"cpu archive bytes")
    command = replace(command, submission=json.dumps(prepared).encode(), source_archives=(archive,))
    records, ref, attempt = owner_at(tmp_path, command)
    bound = LegacyExperiment._observed_command(command, records, ref, attempt)
    expected = "experiment:" + hashlib.sha256(json.dumps([ref.as_dict(), attempt],
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert bound.idempotency_key == expected
    assert not any(item.role == "submission_intent" for item in records.get(ref).artifacts)
    calls = []

    def archived(args, workspace, grace, request, archives, **kwargs):
        assert archives == (archive,) and request["idempotency_key"] == expected
        assert not {"level", "profiler", "counters"}.intersection(request)
        calls.append(request)
        return subprocess.CompletedProcess((), 0, json.dumps({"job_id": "cpu-upload",
            "status": "succeeded", "result": {}}), "")

    monkeypatch.setattr(source_archives, "run_archived", archived)
    monkeypatch.setattr(sandbox, "_find_agate", lambda: "/cpu/agate")
    monkeypatch.setattr(sandbox, "_run_agate_with_cancel_retry", lambda **kwargs: pytest.fail("archive entered raw CLI"))
    assert invoke(bound) == 0 and len(calls) == 1
