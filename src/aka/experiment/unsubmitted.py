"""Explicit original-owner settlement for the diagnosed Legacy local failures.

This host-only interface accepts the original owner's attestation, not an Agent
claim. Absence of a JobRef never supplies the missing submission-boundary proof.
"""
from dataclasses import replace
import hashlib
import json
import re
import uuid

from aka.contracts.execution import ExecutionDiagnostic, ExecutionFacts, JobResult, ResourceIdentity
from aka.contracts.experiment import ExperimentArtifact


def settle(records, ref, attempt_id, *, receipt: bytes, evidence: bytes):
    from .file_records import _encoded, _now
    if not isinstance(receipt, bytes) or not isinstance(evidence, bytes) or not evidence:
        raise TypeError("Original owner receipt and evidence must be retained bytes")
    proof = json.loads(receipt)
    receipt_sha = hashlib.sha256(receipt).hexdigest()
    if (not isinstance(proof, dict) or proof.get("format") != "aka.unsubmitted-owner.v1"
            or proof.get("experiment") != ref.as_dict() or proof.get("attempt_id") != attempt_id
            or proof.get("evidence_sha256") != hashlib.sha256(evidence).hexdigest()
            or proof.get("local_process_tree_terminated") is not True
            or proof.get("submission_boundary_entered") is not False
            or proof.get("uploads_started") is not False
            or not isinstance(proof.get("reason"), str) or not proof["reason"].strip()
            or not isinstance(proof.get("call_order"), list) or not proof["call_order"]
            or any(not isinstance(item, str) or not item.strip() for item in proof["call_order"])
            or proof.get("basis") not in {"legacy-dev-timeout-before-submit", "legacy-admission-rejected"}):
        raise ValueError("Incomplete original-owner pre-submission proof")
    owners = proof.get("original_processes")
    code = proof.get("installed_code_sha256")
    if (not isinstance(owners, list) or not owners or not isinstance(code, dict) or not code
            or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in code.values())):
        raise ValueError("Original process identities and installed code evidence are required")
    from aka.core.processes import process_identity
    for owner in owners:
        if (not isinstance(owner, dict) or type(owner.get("pid")) is not int or owner["pid"] < 1
                or not isinstance(owner.get("start_token"), str) or not owner["start_token"]):
            raise ValueError("Invalid original process identity")
        current = process_identity(owner["pid"])
        if current is not None and current.start_token == owner["start_token"]:
            raise ValueError("Original execution owner is still running")
    original = records.get(ref)
    with records.operation(original.request_id), records._write_lock:
        old = records.get(ref)
        prior = [item for item in old.artifacts if item.role == "unsubmitted-owner-receipt"]
        if prior:
            if (len(prior) == 1 and records.read_artifact(prior[0]) == receipt
                    and old.lifecycle == "terminal" and len(old.attempts) == 1
                    and old.attempts[0].attempt_id == attempt_id
                    and old.attempts[0].disposition == "not_submitted"):
                return old
            raise ValueError("Original settlement receipt differs")
        raw = (records._directory(ref) / "record.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != proof.get("original_record_sha256"):
            raise ValueError("Original Experiment bytes changed")
        if (old.kind != "command" or old.lifecycle == "terminal" or len(old.attempts) != 1
                or len(old.steps) != 1 or old.steps[0].operation != "dev"
                or old.attempts[0].attempt_id != attempt_id or old.attempts[0].disposition != "unknown"):
            raise ValueError("Settlement requires the original single unknown Dev attempt")
        attempt = old.attempts[0]
        result = attempt.result
        if result is not None and (result.job_references or result.attempts or result.exit_code is not None
                or result.stdout or result.file_receipts or result.files
                or result.facts.disposition != "unknown" or result.facts.local_stopped is not True):
            raise ValueError("Original result has unresolved submission or execution evidence")
        if proof["basis"] == "legacy-dev-timeout-before-submit":
            if result is None or result.transport_status != "failed":
                raise ValueError("Original local timeout failure is unavailable")
            stderr = result.stderr
            match = re.search(r"ValueError: Dev timeout (\d+)s exceeds backend limit (\d+)s", stderr)
            if (match is None or int(match[1]) != old.steps[0].timeout or int(match[1]) <= int(match[2])
                    or any(text not in stderr for text in ("_dev_gateway_job_timeout", "_main",
                        "input_transport=inline", "output_transport=none"))
                    or proof.get("stderr_sha256") != hashlib.sha256(stderr.encode()).hexdigest()):
                raise ValueError("Original Dev traceback/transport proof differs")
        elif (result is not None or attempt.failure is None
                or attempt.failure.stage != "execute" or attempt.failure.exception_type not in {"RecoveryRequired", "BudgetDenied"}):
            raise ValueError("Original before-spawn admission rejection is unavailable")
        result = result or JobResult(1, resource=ResourceIdentity("legacy"), hardware=old.steps[0].hardware,
            transport_status="rejected")
        result = replace(result, facts=ExecutionFacts("not_submitted", local_stopped=True,
            cleanup="not_required", basis=("original-owner-confirmed-not-submitted", proof["basis"])),
            diagnostics=(*result.diagnostics, ExecutionDiagnostic("recovery", "original-owner-unsubmitted",
                {"receipt_sha256": receipt_sha})))
        # All evidence is durable before a single atomic record replacement.
        # Existing result artifacts and the exact original envelope are retained.
        artifacts = []
        for role, content in (("unsubmitted-original-record", raw), ("unsubmitted-owner-receipt", receipt),
                ("unsubmitted-owner-evidence", evidence), ("command_result", _encoded(result.as_dict()))):
            artifact = ExperimentArtifact(ref, uuid.uuid4().hex, role, hashlib.sha256(content).hexdigest(), len(content))
            records._artifact_storage.publish(records._directory(ref) / "artifacts" / artifact.artifact_id,
                content, artifact.sha256)
            artifacts.append(artifact)
        return records._save(replace(old, attempts=(replace(attempt, disposition="not_submitted", result=result),),
            artifacts=(*old.artifacts, *artifacts), lifecycle="terminal", phase="complete", outcome="inconclusive",
            record_version=old.record_version + 1, updated_at=_now()))
