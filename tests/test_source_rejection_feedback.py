"""SDK admission causes reach Coder feedback without publishing private text."""
from dataclasses import replace
import hashlib
import json
import os
import socket
from types import SimpleNamespace

import pytest

from aka.contracts.execution import ExecutionDiagnostic, ExecutionFacts, JobResult
from aka.contracts.execution_recovery import ExecutionCause, JobReference
from aka.evaluation.tool_results import canonical_result, with_execution_error
from aka.experiment.file_records import FileExperimentRecords
from aka.experiment.runner import LegacyExperiment
from aka.workflow.light.measurements import SharedMeasurements
import test_live_eval_recovery as live


RULE = "candidate: Forbidden name: getattr"
PRIVATE = "cpu-private-source-error-sentinel"


def envelope(errors=None):
    return {"error_class": "validation", "reason": "invalid_source",
        "message": "source validation failed", "trace_id": PRIVATE,
        "details": {"validation_errors": [RULE] * 3 if errors is None else errors,
            "signed_url": "https://cpu.invalid/input?Signature=" + PRIVATE,
            "env_vars": {"AGATE_SK": PRIVATE}, "user_action": PRIVATE},
        "unrecognized_private_field": PRIVATE}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("source feedback CPU tests must not make network calls")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


def canonical(receipt):
    return canonical_result(with_execution_error({"exit_code": receipt.returncode,
        "stdout": receipt.stdout, "stderr": receipt.stderr}, receipt), {"operation": "evaluate"})


def assert_private_absent(value):
    text = json.dumps(value)
    assert all(marker not in text for marker in
        (PRIVATE, "signed_url", "Signature=", "env_vars", "AGATE_SK", "original_sdk_error", "validation_errors"))


@pytest.mark.parametrize("errors,repairable", [
    ([RULE] * 3, True),
    ([RULE, "candidate: Blocked import: importlib"], True),
    (["reference.input_py: Forbidden name: getattr"], False),
    ([RULE, "reference: Forbidden name: getattr"], False),
    ([RULE, "env_vars: " + PRIVATE], False),
    (["worker environment: " + PRIVATE], False),
    (["candidate: SyntaxError: " + PRIVATE], True),
    ([RULE, "candidate: Forbidden name: " + PRIVATE], True),
    (["candidate: " + PRIVATE + "\nreference: Forbidden name: getattr"], False),
])
def test_real_legacy_sdk_record_to_coder_projection(tmp_path, monkeypatch, errors, repairable):
    # Patch only the existing CPU fixture's native HTTP response. Legacy launch,
    # SDK signing/serialization, owned fact pipe and both record owners are real.
    needle = '        key = payload["idempotency_key"]\n'
    replacement = needle + '''        if state["mode"] == "source-rejection":
            state_path.write_text(json.dumps(state))
            raise GatewayError(400, state["rejection"])
'''
    assert needle in live.CHILD
    monkeypatch.setattr(live, "CHILD", live.CHILD.replace(needle, replacement))
    f = live.fixture(tmp_path, monkeypatch, mode="source-rejection")
    # Keep fixture paths in its local launcher, so the actual Eval worker env is
    # empty just as in the rejected production request.
    child = tmp_path / "cpu_launcher.py"
    child.write_text(child.read_text().replace('import json, os, sys\n',
        'import json, os, sys\nsys.path[:0] = ' + repr(f.command.environment["PYTHONPATH"].split(os.pathsep)) + '\n')
        .replace('os.environ["CPU_STATE"]', repr(str(f.state_path))))
    service = live.read_service(f)
    service["rejection"] = envelope(errors)
    f.state_path.write_text(json.dumps(service))
    records = FileExperimentRecords(tmp_path / "pipeline", "cpu-source-feedback")
    experiment = LegacyExperiment(f.executor, records=records, purpose="formal")
    prepared_command = json.loads(f.command.submission)
    prepared_command["request"]["env_vars"] = {}
    from aka.execution.configuration import transport_environment_key
    command = replace(f.command, environment={key: value for key, value in f.command.environment.items()
        if transport_environment_key(key)}, submission=json.dumps(prepared_command).encode())
    with f.state.owned(), f.executor.dispatch_scope(admit=f.state.admit_execution, observe=f.state.observe_execution):
        f.state.action("implement")
        record = experiment.run_recorded(command, request_id="source-rejected")
    assert record.lifecycle == "terminal" and record.phase == "complete"
    assert len(record.attempts) == len(f.launches) == f.state.data["counts"]["dispatches"] == 1
    assert all(call["state"] == "settled" for call in f.state.data["calls"].values())
    service = live.read_service(f)
    call, = service["calls"]
    assert (call["method"], call["path"]) == ("POST", "/v1/jobs/eval") and not service["jobs"]
    binding_artifact, = [item for item in record.artifacts if item.role == "submission_intent"]
    binding = json.loads(records.read_artifact(binding_artifact))
    prepared_artifact, = [item for item in record.artifacts if item.sha256 == binding["submission_sha256"]]
    prepared = json.loads(records.read_artifact(prepared_artifact))
    expected = {**prepared["request"], "idempotency_key": binding["idempotency_key"]}
    assert call["payload"] == expected
    assert call["payload"]["env_vars"] == {}

    fresh = FileExperimentRecords(tmp_path / "pipeline", "cpu-source-feedback", create=False)
    receipt = fresh.command_result(record.ref)
    assert receipt.transport_status == "rejected" and receipt.facts.disposition == "not_submitted"
    assert not receipt.job_references and not receipt.stdout and receipt.stderr == "HTTP 400\n"
    assert receipt.attempts[0].idempotency_key == binding["idempotency_key"]
    cause, = receipt.causes
    assert (cause.category, cause.code) == ("validation", "invalid_source")
    assert cause.details["original_sdk_error"]["payload"]["trace_id"] == PRIVATE
    before = {item.artifact_id: hashlib.sha256(fresh.read_artifact(item)).hexdigest() for item in record.artifacts}
    measurements = SharedMeasurements(factory=None, executor=None, evaluator=None, namespace_workspace=tmp_path,
        evaluation_input=None, workspace_provider=SimpleNamespace(
            source_identity=lambda source: hashlib.sha256(source.files["kernel.py"]).hexdigest()),
        command_evaluator=None)
    response = measurements._response(fresh, fresh.get(record.ref))
    agent = response["agent_result"]
    error = agent["result"]["error"]
    assert error["candidate_repairable"] is repairable
    assert agent["status"] != "completed" and agent["result"]["correct"] is None
    assert agent["result"]["latency_us_geomean"] is None
    assert agent["result"]["latency_us_by_shape"] == {}
    assert agent["result"]["outcome"] == "not_submitted"
    if repairable:
        assert error["code"] == "candidate_source_rejected" and error["error_class"] == "code"
        assert error["job_submitted"] is False
        known = {RULE, "candidate: Blocked import: importlib"}
        assert error["violations"] == list(dict.fromkeys(item.removeprefix("candidate: ") for item in errors if item in known))
        assert "validate" in agent["result"]["next_action"]
        if not error["violations"]:
            assert "Repair the candidate source" in agent["result"]["next_action"]
    else:
        assert error["code"] == "gateway-http-rejection" and error["error_class"] == "unknown"
    assert_private_absent(response)
    assert before == {item.artifact_id: hashlib.sha256(fresh.read_artifact(item)).hexdigest() for item in record.artifacts}


def receipt_for(payload=None):
    return JobResult(2, stderr="HTTP 400\n", transport_status="rejected",
        facts=ExecutionFacts("not_submitted", local_stopped=True),
        diagnostics=(ExecutionDiagnostic("request", "gateway-http-rejection", {"http_status": 400}),),
        causes=(ExecutionCause("validation", "invalid_source", PRIVATE,
            {"original_sdk_error": {"status": 400, "payload": envelope() if payload is None else payload}}),))


@pytest.mark.parametrize("change", [
    "status", "class", "code", "message", "missing-errors", "empty-errors", "string-errors",
    "object-error", "oversized-errors", "empty-detail", "oversized-detail", "nested-envelope", "unstructured-cause",
    "cause-class", "cause-code", "child-cause", "mixed-cause", "no-http-fact", "http-401",
    "authentication", "reference-fact", "unknown-disposition", "terminal-disposition", "job-reference",
])
def test_unknown_or_conflicting_evidence_never_becomes_candidate_repairable(change):
    result = receipt_for()
    payload = envelope()
    original = {"status": 400, "payload": payload}
    cause = result.causes[0]
    if change == "status":
        original["status"] = 401
    elif change in {"class", "code", "message"}:
        payload[{"class": "error_class", "code": "reason", "message": "message"}[change]] = PRIVATE
    elif change == "missing-errors":
        payload["details"].pop("validation_errors")
    elif change in {"empty-errors", "string-errors", "object-error", "oversized-errors", "empty-detail", "oversized-detail"}:
        payload["details"]["validation_errors"] = {"empty-errors": [], "string-errors": RULE,
            "object-error": [{"candidate": PRIVATE}], "oversized-errors": [RULE] * 4097,
            "empty-detail": ["candidate: "], "oversized-detail": ["candidate: " + PRIVATE * 4096]}[change]
    elif change == "nested-envelope":
        original["payload"] = {"error": payload}
    elif change == "unstructured-cause":
        original = None
    cause = replace(cause, details={"original_sdk_error": original})
    if change in {"cause-class", "cause-code"}:
        cause = replace(cause, **{"category" if change == "cause-class" else "code": PRIVATE})
    elif change == "child-cause":
        cause = replace(cause, causes=(ExecutionCause("validation", "invalid_source", PRIVATE),))
    result = replace(result, causes=(cause,))
    if change == "mixed-cause":
        result = replace(result, causes=(*result.causes, ExecutionCause("validation", "invalid_source", PRIVATE)))
    elif change == "no-http-fact":
        result = replace(result, diagnostics=())
    elif change == "http-401":
        result = replace(result, diagnostics=(ExecutionDiagnostic("request", "gateway-http-rejection", {"http_status": 401}),))
    elif change in {"authentication", "reference-fact"}:
        extra = ExecutionDiagnostic("authentication", "denied") if change == "authentication" else (
            ExecutionDiagnostic("request", "invalid_source", {"reference_rejected": True}))
        result = replace(result, diagnostics=(*result.diagnostics, extra))
    elif change.endswith("-disposition"):
        result = replace(result, facts=replace(result.facts, disposition=change.removesuffix("-disposition")))
    elif change == "job-reference":
        result = replace(result, job_references=(JobReference("gateway", "cpu-original-job"),))
    public = canonical(result)
    assert public["result"]["error"]["candidate_repairable"] is False
    assert public["result"]["error"]["error_class"] == "unknown"
    assert_private_absent(public)


def test_cli_candidate_diagnostics_still_use_the_existing_public_protocol():
    result = replace(receipt_for(), causes=(),
        stderr="agate: [400] source validation failed\n  - candidate: Forbidden name: getattr\n",
        diagnostics=(ExecutionDiagnostic("request", "empty-stdout", {"http_status": 400,
            "source_validation_failed": True, "candidate_rejected": True, "reference_rejected": False}),))
    error = canonical(result)["result"]["error"]
    assert error["candidate_repairable"] is True and error["violations"] == ["Forbidden name: getattr"]
