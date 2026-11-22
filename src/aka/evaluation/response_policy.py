"""Stable classifications for Agate responses Runtime intentionally understands."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import re

from aka.contracts.execution import JobResult

SOURCE_VIOLATION = re.compile(
    r"(?:Blocked import|Forbidden attribute access|Forbidden name): [A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
    r"|Forbidden string literal matching: [A-Za-z_]\w*"
)
INFORMATION_CATEGORIES = frozenset({"recovery", "observation", "budget"})
# Fixed public rules already recognized by the gateway protocol adapter. Never
# publish arbitrary identifiers or text from the private SDK validation payload.
_SDK_SOURCE_RULES = {
    "candidate: Forbidden name: getattr": "Forbidden name: getattr",
    "candidate: Blocked import: importlib": "Blocked import: importlib",
}


def _sdk_source_violations(result: JobResult) -> list[str] | None:
    """None means unrecognized; an empty list is a candidate-only private rule."""
    if not result.causes or not any(item.category == "request" and item.details.get("http_status") == 400
                                    for item in result.diagnostics):
        return None
    violations = []
    for cause in result.causes:
        original = cause.details.get("original_sdk_error")
        if (cause.category != "validation" or cause.code != "invalid_source" or cause.causes
                or not isinstance(original, Mapping) or original.get("status") != 400):
            return None
        payload = original.get("payload")
        if (not isinstance(payload, Mapping) or payload.get("error_class") != "validation"
                or payload.get("reason") != "invalid_source"
                or payload.get("message") != "source validation failed"):
            return None
        details = payload.get("details")
        errors = details.get("validation_errors") if isinstance(details, Mapping) else None
        if not isinstance(errors, (list, tuple)) or not 0 < len(errors) <= 4096:
            return None
        for error in errors:
            # SDK field attribution proves what to repair without publishing
            # its arbitrary detail. Mixed or ambiguous fields veto that claim.
            if (not isinstance(error, str) or not error.startswith("candidate: ")
                    or not len("candidate: ") < len(error) <= 4096 or "\n" in error or "\r" in error):
                return None
            violation = _SDK_SOURCE_RULES.get(error)
            if violation is not None and violation not in violations:
                violations.append(violation)
    return violations


def execution_error(result: JobResult) -> dict[str, Any] | None:
    """Interpret captured execution failure facts without inferring dispatch from text."""
    diagnostics = tuple(item for item in result.diagnostics if item.category not in INFORMATION_CATEGORIES)
    if result.returncode == 0 and not diagnostics and result.transport_status in {"succeeded", "failed", "unknown"}:
        return None
    admission_rejected = (
        result.facts.disposition == "not_submitted" and not result.job_references
        and not any(item.category == "authentication" or item.details.get("reference_rejected")
                    for item in result.diagnostics)
    )
    sdk_violations = _sdk_source_violations(result) if admission_rejected else None
    candidate_rejected = admission_rejected and (
        sdk_violations is not None or any(item.details.get("http_status") == 400
            and item.details.get("source_validation_failed") is True
            and item.details.get("candidate_rejected") is True
            and not item.details.get("reference_rejected")
            for item in result.diagnostics)
    )
    if candidate_rejected:
        violations = list(sdk_violations or ())
        # SDK exception strings and retained causes stay private. The older CLI
        # path has its own public, field-qualified stderr protocol.
        for line in (() if sdk_violations is not None else result.stderr.splitlines()):
            match = re.match(r"\s*- candidate:\s*(.*)", line)
            if match:
                value = re.sub(r"\s*\(trace_id=[^)]+\)\s*$", "", match[1])
                if SOURCE_VIOLATION.fullmatch(value) and value not in violations:
                    violations.append(value)
        return {"status": "failed", "outcome": "not_submitted", "error": {
            "code": "candidate_source_rejected", "error_class": "code",
            "message": "Agate rejected candidate source before GPU submission.",
            "reason": "source_validation_failed", "job_submitted": False,
            "candidate_repairable": True, "violations": violations},
            "next_action": ("Replace the reported forbidden names, imports or attribute accesses in the candidate. "
                if violations else "Repair the candidate source to satisfy the gateway source policy. ") +
                "Use the documented APIs for the pinned runtime version, then validate the repaired source. "
                "Do not bypass source validation through Dev."}
    primary = result.primary_diagnostic if result.primary_diagnostic in diagnostics else None
    primary = primary or (diagnostics[0] if diagnostics else None)
    message = result.stderr.strip() or result.stdout.strip()
    if not message:
        message = "; ".join(f"{item.category}/{item.code}: {dict(item.details)}"
            for item in diagnostics)
    if not message:
        message = f"Execution failed: status={result.transport_status}; exit={result.returncode}"
    return {"status": "failed", "outcome": result.facts.disposition, "error": {
        "code": primary.code if primary is not None else "execution_failed",
        "error_class": "unknown", "message": message,
        "candidate_repairable": False,
        "execution_status": result.transport_status,
        "diagnostics": [{"category": item.category, "code": item.code,
            "details": dict(item.details)} for item in diagnostics]},
        "next_action": "Inspect the returned error and original execution receipt. "
            "An unknown execution outcome requires original-job reconciliation; do not resubmit blindly."}


KNOWN_GATEWAY_ERROR_CLASSES = frozenset({
    "candidate",
    "code",
    "compilation",
    "correctness",
    "infra",
    "validation",
})


def known_gateway_error(value: dict[str, Any]) -> bool:
    """Whether Runtime has an intentional recovery policy for this Agate error."""
    if value.get("outcome") == "unknown" or value.get("status") in {
        "cancelled",
        "canceled",
    }:
        return True
    error = value.get("error")
    return (
        isinstance(error, dict)
        and error.get("error_class") in KNOWN_GATEWAY_ERROR_CLASSES
    )
