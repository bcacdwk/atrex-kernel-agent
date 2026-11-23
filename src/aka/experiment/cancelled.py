"""Terminal observation of a cancelled, recorded live Eval; never a replay."""
from dataclasses import replace
import hashlib
import json
import uuid

from aka.contracts.execution import ExecutionDiagnostic
from aka.contracts.experiment import ExperimentArtifact, ExperimentRecordError
from aka.execution.live_eval import digest, encoded


def association(records, record):
    """Validate the original immutable input and the complete physical history."""
    return _association(records, record)[0]


def _association(records, record):
    def reject():
        raise ExperimentRecordError("cancelled_eval_identity_mismatch", record.ref)
    if (record.kind != "command" or len(record.steps) != 1 or len(record.attempts) != 1
            or record.steps[0].operation != "run"):
        reject()
    attempt, = record.attempts
    result = attempt.result
    if (result is None or result.transport_status != "cancelled"
            or result.facts.cancellation_requested is not True or result.facts.local_stopped is not True
            or len(result.job_references) != 1 or not result.attempts):
        reject()
    documents = {}
    for artifact in record.artifacts:
        if artifact.role in {"submission_intent", "live_eval_association", "live_eval_physical", "evaluation_input"}:
            documents.setdefault(artifact.role, []).append((artifact, json.loads(records.read_artifact(artifact))))
    intents = [value for _, value in documents.get("submission_intent", ())
               if value.get("attempt_id") == attempt.attempt_id]
    links = [value for _, value in documents.get("live_eval_association", ())
             if value.get("attempt_id") == attempt.attempt_id]
    if len(intents) != 1 or len(links) != 1:
        reject()
    intent, link = intents[0], links[0]
    reference, = result.job_references
    physical_ids = [item.attempt_id for item in result.attempts if item.operation == "run"]
    if (link.get("job_reference") != result.as_dict()["job_references"][0]
            or link.get("physical_attempts") != physical_ids or len(set(physical_ids)) != len(physical_ids)
            or link.get("terminal") is not False or not physical_ids
            or intent.get("idempotency_key") != reference.idempotency_key
            or intent.get("operation") != "run" or intent.get("timeout") != record.steps[0].timeout
            or link.get("idempotency_key") != reference.idempotency_key
            or link.get("request_identity") != reference.request_identity):
        reject()
    inputs = [value for artifact, value in documents.get("evaluation_input", ())
              if artifact.sha256 == intent.get("submission_sha256")]
    if len(inputs) != 1 or inputs[0].get("kind") != "run" or inputs[0].get("schema_version") != 1:
        reject()
    from aka.execution.gateway.requests import operation_request
    request = operation_request(inputs[0]["request"], "run")
    request["idempotency_key"] = reference.idempotency_key
    request.setdefault("options", {})["timeout_s"] = record.steps[0].timeout
    if digest(request) != reference.request_identity:
        reject()
    sources = [item for item in record.artifacts if item.role == "candidate-source"]
    if len(sources) != 1 or records.read_source(sources[0]).files.get("kernel.py") != request["candidate"].encode():
        reject()
    identity = {key: link[key] for key in ("attempt_id", "idempotency_key", "request_identity",
                                         "route_identity", "namespace_identity")}
    physical = documents.get("live_eval_physical", ())
    if len(physical) != len(physical_ids):
        reject()
    for (_, value), physical_id in zip(physical, physical_ids):
        raw = value.get("result", {})
        receipts = raw.get("attempts", ())
        if (value.get("identity") != identity or len(receipts) != 1
                or raw.get("facts", {}).get("local_stopped") is not True
                or receipts[0].get("attempt_id") != physical_id
                or receipts[0].get("operation") != "run"
                or receipts[0].get("idempotency_key") != reference.idempotency_key
                or receipts[0].get("request_identity") != reference.request_identity
                or receipts[0].get("resource") not in ({"backend": "gateway", "identifier": None},
                    {"backend": "gateway", "identifier": reference.identifier})
                or any(item != link["job_reference"] for item in raw.get("job_references", ()))):
            reject()
    receipt = encoded(link).decode()
    for item in result.attempts:
        if item.operation == "query":
            continue
        if (item.operation != "run" or item.idempotency_key != reference.idempotency_key
                or item.request_identity != reference.request_identity or item.resource is None
                or item.resource.identifier != reference.identifier or item.resource.backend != "gateway"
                or not any(d.code == "live-eval-association" and d.category == "recovery"
                           and d.details.get("receipt") == receipt for d in item.diagnostics)):
            reject()
    return link, request


def _cancelled_and_clean(job, observed):
    # Reuse the parsed remote-termination fact and backend cause details. A
    # terminal status cannot override conflicting or ambiguous cleanup evidence.
    sources = [job]
    causes = list(observed.causes)
    while causes:
        cause = causes.pop()
        sources.append(cause.details)
        causes.extend(cause.causes)
    return (job.get("status") == "cancelled" and observed.facts.remote_terminated is not False
        and job.get("cleanup_pending") is False and job.get("execution_quarantined") is False
        and job.get("execution_state") in ("stopped", "not_submitted")
        and all(source.get("cleanup_pending", False) is False
            and source.get("execution_quarantined", False) is False
            and (source.get("remote_terminated") is None or source.get("remote_terminated") is True)
            and source.get("execution_state", job["execution_state"]) == job["execution_state"]
            for source in sources))


def _echo_matches(job, request):
    # Slim Job views may omit these optional echoes (or return null). Identity
    # comes from the immutable request/association and the bound Job GET; an
    # echo that is present must not contradict that evidence.
    solution = job.get("solution")
    if solution is not None:
        descriptor = solution.get("source") if isinstance(solution, dict) else None
        if (not isinstance(descriptor, dict) or descriptor.get("type") != "inline"
                or descriptor.get("files") != {"kernel.py": hashlib.sha256(request["candidate"].encode()).hexdigest()}):
            return False
    summary = job.get("request_digest")
    if summary is not None and not isinstance(summary, dict):
        return False
    summary = summary or {}
    spec = request.get("spec", {})
    # The gateway may prefix the requested device label with its vendor name.
    def device(value):
        return value.removeprefix("NVIDIA ").strip().casefold() if isinstance(value, str) else None
    if "gpu" in summary and (not isinstance(summary["gpu"], str)
            or device(summary["gpu"]) not in {device(value) for value in spec.get("target_hardware", ())}):
        return False
    expected = {"kind": "eval", "num_gpus": spec.get("num_gpus", 1),
                "timeout_s": request["options"]["timeout_s"]}
    for source, fields in ((summary, expected), (job, {"num_gpus": expected["num_gpus"]})):
        if any(name in source and (type(source[name]) is not type(value) or source[name] != value)
               for name, value in fields.items()):
            return False
    return "mode" not in job or "mode" not in request or job["mode"] == request["mode"]


def reconcile(records, executor, ref, *, timeout):
    from .file_records import _encoded, _now
    original = records.get(ref)
    # The original native operation lock proves no tool owner can still write.
    with records.operation(original.request_id), records._write_lock:
        old = records.get(ref)
        link, request = _association(records, old)
        prior = [item for item in old.artifacts if item.role == "cancelled-eval-receipt"]
        if prior:
            if (len(prior) != 1 or old.lifecycle != "terminal" or old.outcome != "inconclusive"
                    or old.attempts[0].disposition != "terminal"):
                raise ExperimentRecordError("cancelled_eval_receipt_mismatch", ref)
            proof = json.loads(records.read_artifact(prior[0]))
            if proof.get("association") != link:
                raise ExperimentRecordError("cancelled_eval_receipt_mismatch", ref)
            return old
        if old.lifecycle == "terminal" or old.attempts[0].disposition != "unknown":
            raise ExperimentRecordError("cancelled_eval_not_pending", ref)
        query = getattr(executor, "query_recorded_eval", None)
        if not callable(query):
            raise ExperimentRecordError("cancelled_eval_query_unavailable", ref)
        observed = query(link, timeout=timeout)
        attempt, = old.attempts
        result = attempt.result
        try:
            job = json.loads(observed.stdout)
        except (TypeError, ValueError):
            job = {}
        if not isinstance(job, dict):
            job = {}
        if (observed.job_references != result.job_references or observed.facts.disposition != "terminal"
                or observed.facts.local_stopped is not True or observed.transport_status != "cancelled"
                or not observed.attempts or any(item.operation != "query" or item.disposition != "terminal"
                    or item.attempt_id is not None or item.resource != result.attempts[-1].resource
                    or item.idempotency_key != link["idempotency_key"]
                    or item.request_identity != link["request_identity"] for item in observed.attempts)
                or job.get("job_id") != result.job_references[0].identifier or job.get("kind") != "eval"
                or not _cancelled_and_clean(job, observed)
                or job.get("idempotency_key", link["idempotency_key"]) != link["idempotency_key"]
                or not _echo_matches(job, request)):
            raise ExperimentRecordError("experiment_pending", ref)
        raw = (records._directory(ref) / "record.json").read_bytes()
        proof = encoded({"format": "aka.cancelled-eval.v1", "experiment": ref.as_dict(),
            "original_record_sha256": hashlib.sha256(raw).hexdigest(), "association": link,
            "query_sha256": hashlib.sha256(_encoded(observed.as_dict())).hexdigest()})
        result = replace(result, facts=replace(observed.facts, cancellation_requested=True),
            attempts=(*result.attempts, *observed.attempts),
            diagnostics=(*result.diagnostics, *observed.diagnostics,
                ExecutionDiagnostic("recovery", "cancelled-eval-settled", {"receipt_sha256": hashlib.sha256(proof).hexdigest()})),
            causes=(*result.causes, *observed.causes))
        artifacts = []
        for role, content in (("cancelled-eval-original-record", raw), ("cancelled-eval-query", _encoded(observed.as_dict())),
                              ("cancelled-eval-receipt", proof), ("command_result", _encoded(result.as_dict()))):
            artifact = ExperimentArtifact(ref, uuid.uuid4().hex, role, hashlib.sha256(content).hexdigest(), len(content))
            records._artifact_storage.publish(records._directory(ref) / "artifacts" / artifact.artifact_id, content, artifact.sha256)
            artifacts.append(artifact)
        return records._save(replace(old, attempts=(replace(attempt, disposition="terminal", result=result),),
            artifacts=(*old.artifacts, *artifacts), lifecycle="terminal", phase="complete", outcome="inconclusive",
            record_version=old.record_version + 1, updated_at=_now()))
