"""Bounded continuation of one still-live, recorded inline native submission.

The persisted links are evidence, never permission to replay after owner exit.
Only the original open dispatch scope can spend another native-call permit.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import time
from threading import Event

from aka.contracts.execution import ExecutionDiagnostic, ResourceIdentity, UnsupportedExecution
from aka.contracts.execution_recovery import InfrastructureRetryPolicy


WIRE_PATH = "/v1/jobs/eval"
LIVE_ENV = "AKA_EXECUTION_LIVE_EVAL"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


class RecordedEval:
    """One-use authority attached only while publishing a new owner intent."""
    def __init__(self, binding, archive):
        self.binding = dict(binding)
        self.archive = archive
        self.claimed = False


class _LiveCancellation:
    def __init__(self, live):
        self.live = live

    def is_set(self):
        parent = self.live.command.cancel_event
        return self.live.remaining() <= 0 or parent is not None and parent.is_set()

    def wait(self, timeout):
        end = time.monotonic() + timeout
        while not self.is_set():
            remaining = min(end - time.monotonic(), self.live.remaining())
            if remaining <= 0:
                return self.is_set()
            Event().wait(min(.05, remaining))
        return True


class LiveEval:
    def __init__(self, owner, command, request, scope, transport, *, wait_timeout, unchanged):
        deadline = time.monotonic() + wait_timeout if wait_timeout is not None else float("inf")
        if owner.claimed:
            raise UnsupportedExecution("recorded Eval continuation already left its live owner")
        binding = owner.binding
        if (binding["idempotency_key"] != command.idempotency_key
                or binding["submission_sha256"] != hashlib.sha256(command.submission).hexdigest()
                or binding["environment_sha256"] != digest(dict(command.environment))
                or binding["timeout"] != command.timeout
                or binding["operation"] != command.operation):
            raise UnsupportedExecution("recorded Eval continuation identity changed")
        environment = request.environment
        if (not transport.endpoint or transport.profile or request.ssh
                or environment.get("AGATE_TOKEN")
                or not environment.get("AGATE_AK") or not environment.get("AGATE_SK")):
            raise UnsupportedExecution("live Eval requires a verified explicit URL and frozen AK/SK")
        # The simple explicit connection has no mutable named/routed profile.
        from atrex_gateway_client import Client  # Fail before admission if unavailable.
        del Client
        from .gateway.requests import operation_request, prepared_operation, submission_path
        prepared = json.loads(request.submission)
        kind = prepared_operation(prepared, command.operation)
        wire = operation_request(prepared["request"], kind, ssh=False)
        if kind in {"run", "profile"}:
            wire.setdefault("options", {})["timeout_s"] = command.timeout
        self.operation = kind
        path = submission_path(kind)
        self.identity = {"attempt_id": binding["attempt_id"], "idempotency_key": command.idempotency_key,
            "request_identity": digest(wire), "route_identity": digest([transport.endpoint, path, "", ""]),
            "namespace_identity": digest(environment["AGATE_AK"])}
        if kind != "run":
            # Keep the exact final auxiliary body with its original owner before
            # any POST, independently of launcher output and the receipt pipe.
            owner.archive(encoded({"identity": self.identity, "request": wire}), "live_eval_submission")
        self.owner, self.command, self.scope = owner, command, scope
        self.endpoint = transport.endpoint
        self.unchanged = unchanged
        self.deadline = deadline
        self.policy = InfrastructureRetryPolicy()
        self.recovery_deadline = None
        self.permits, self.completed = [], []
        self.state = None
        self.active = True
        self.association = None
        self.latest = None
        self.cancel_event = _LiveCancellation(self)
        self.owner.claimed = True

    def validate(self, command=None):
        if (not self.active or self.scope.closed or command is not None and command is not self.command
                or not self.unchanged()
                or self.command.idempotency_key != self.owner.binding["idempotency_key"]
                or self.command.operation != self.owner.binding["operation"]
                or digest(dict(self.command.environment)) != self.owner.binding["environment_sha256"]
                or self.command.timeout != self.owner.binding["timeout"]
                or hashlib.sha256(self.command.submission).hexdigest() != self.owner.binding["submission_sha256"]):
            raise UnsupportedExecution("live Eval owner, scope or frozen identity changed")
        if self.command.cancel_event is not None and self.command.cancel_event.is_set():
            from concurrent.futures import CancelledError
            raise CancelledError("live Eval owner cancelled")
        if self.remaining() <= 0:
            raise TimeoutError("original Eval client deadline exhausted")

    def admit(self, state, command):
        self.validate(command)
        if self.state is None:
            self.state = state
        if self.state is not state:
            raise UnsupportedExecution("live Eval cannot change its Schema owner")
        for permit in self.permits:
            call = state.data["calls"].get(permit, {})
            if (call.get("live_eval") != self.identity
                    or call.get("action") != (state.data["action"] or {}).get("id")):
                raise UnsupportedExecution("live Eval permit identity changed")
        return frozenset(self.permits)

    def remaining(self):
        return max(0, self.absolute_deadline() - time.monotonic())

    def absolute_deadline(self):
        return min(self.deadline, self.recovery_deadline or float("inf"))

    def launch_binding(self):
        deadline = self.absolute_deadline()
        return {**self.identity, "client_deadline": None if deadline == float("inf") else deadline}

    def retry(self, result):
        if (len(self.completed) > len(self.policy.delays)
                or result.transport_status != "infrastructure" or result.facts.disposition != "unknown"
                or result.facts.local_stopped is not True or result.facts.cancellation_requested
                or result.facts.cleanup == "failed" or result.job_references or not result.attempts):
            return False
        if any(not self.matches(item) or item.disposition != "unknown" for item in result.attempts):
            return False
        diagnostics = (*result.diagnostics, *(item for attempt in result.attempts for item in attempt.diagnostics))
        if not any(item.category == "infrastructure" and item.details.get("http_status") == 503
                   for item in diagnostics):
            return False
        if any(item.category in {"authentication", "request", "cancelled", "timeout", "protocol", "cleanup"}
               for item in diagnostics):
            return False
        for item in diagnostics:
            if item.code == "gateway-policy":
                values = json.loads(item.details.get("payload", "{}"))
                if (values.get("replay_policy") == "never" or values.get("execution_state") == "unknown"
                        or any(value == "never" for key, value in values.items() if key.endswith("retry_advice"))):
                    return False
        self.validate()
        if self.recovery_deadline is None:
            self.recovery_deadline = time.monotonic() + self.policy.max_elapsed_s
        delay = self.policy.delays[len(self.completed) - 1]
        if delay >= self.remaining():
            return False
        if self.cancel_event.wait(delay):
            self.validate()
            return False
        self.validate()
        return self.remaining() > 0

    def matches(self, attempt):
        return (attempt.attempt_id and attempt.operation == self.operation
                and attempt.idempotency_key == self.identity["idempotency_key"]
                and attempt.request_identity == self.identity["request_identity"])

    def project(self, result, *, final=False):
        previous = tuple(attempt for item in self.completed for attempt in item.attempts)
        attempts = (*previous, *result.attempts)
        references = tuple(dict.fromkeys(ref for item in (*self.completed, result) for ref in item.job_references))
        association = None
        if len(references) == 1 and attempts and all(self.matches(item) for item in attempts):
            ref = references[0]
            if (ref.operation == self.operation and ref.endpoint == self.endpoint and not ref.profile
                    and ref.request_identity == self.identity["request_identity"]
                    and ref.idempotency_key == self.identity["idempotency_key"]):
                association = {**self.identity, "job_reference": result.as_dict()["job_references"][0]
                    if result.job_references else self.completed[-1].as_dict()["job_references"][0],
                    "physical_attempts": [item.attempt_id for item in attempts]}
        terminal = bool(association and final and result.facts.local_stopped is True
            and result.facts.disposition == "terminal"
            and all(item.disposition == "terminal" for item in result.attempts))
        diagnostics = tuple(item for value in (*self.completed, result) for item in value.diagnostics)
        if association:
            receipt = ExecutionDiagnostic("recovery", "live-eval-association", {
                "receipt": encoded({**association, "terminal": terminal}).decode()})
            if final:
                self.owner.archive(encoded({**association, "terminal": terminal}), "live_eval_association")
                self.association = {**association, "terminal": terminal}
            attempts = tuple(replace(item, resource=ResourceIdentity(ref.backend, ref.identifier),
                disposition="terminal" if terminal else item.disposition,
                diagnostics=(*item.diagnostics, receipt)) for item in attempts)
            diagnostics = (*diagnostics, receipt)
        unresolved = any(item.disposition == "unknown" for item in attempts)
        facts = replace(result.facts, disposition="unknown") if unresolved else result.facts
        return replace(result, facts=facts, attempts=attempts, diagnostics=diagnostics,
            causes=tuple(item for value in (*self.completed, result) for item in value.causes),
            stderr="\n".join(value.stderr for value in (*self.completed, result) if value.stderr),
            job_references=references)

    def finish(self, result):
        # Preserve exact physical evidence before any causal projection replaces it.
        # Keep the current raw receipt even if archival/observation subsequently
        # fails: an exception must not replace its known JobRef with old facts.
        if self.recovery_deadline is None and result.transport_status == "infrastructure":
            # Artifact publication and owner/root observation spend this same
            # elapsed allowance; they must not postpone its starting clock.
            self.recovery_deadline = time.monotonic() + self.policy.max_elapsed_s
        self.latest = self.project(result)
        self.owner.archive(encoded({"identity": self.identity, "permit": self.permits[-1],
            "result": result.as_dict()}), "live_eval_physical")
        projected = self.project(result, final=True)
        self.completed.append(result)
        self.latest = projected
        return projected


def sdk_http(environment, endpoint, *, deadline=None, transport_retries=2, operation="run"):
    """Use SDK signing/serialization, with no profile/config or legacy POST fallback."""
    from atrex_gateway_client import Client
    from atrex_gateway_client.auth import AkSkAuth
    from atrex_gateway_client.errors import GatewayError
    from .gateway.requests import submission_path
    import urllib.request
    path_for_submission = submission_path(operation)

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    class OriginalRouteClient(Client):
        def _request(self, method, path, body=None, timeout=None):
            # Submission cannot outlive the owner deadline. Cancellation and
            # its confirming GET retain their separate bounded cleanup budget.
            if deadline is not None and method == "POST" and path == path_for_submission:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise GatewayError(408, {"reason": "original_client_deadline_exhausted"})
                timeout = min(timeout or self.timeout, remaining)
            if method == "POST" and path != path_for_submission and not path.endswith("/cancel"):
                raise self.original_route_error
            try:
                return super()._request(method, path, body, timeout)
            except GatewayError as error:
                # If submit_job attempts its compatibility fallback, return the
                # actual original-route rejection without issuing another POST.
                self.original_route_error = error
                raise

    client = OriginalRouteClient(endpoint, auth=AkSkAuth(environment["AGATE_AK"], environment["AGATE_SK"]),
                                 transport_retries=transport_retries)
    if not hasattr(client, "_environment_opener") or not hasattr(client, "_connection"):
        raise UnsupportedExecution("SDK lacks the required frozen-connection adapter")
    client.original_route_error = GatewayError(409, {"reason": "original_submission_route_required"})
    # The SDK only installs its no-redirect opener for routed profiles. This
    # exact-origin continuation also refuses redirects for the plain URL case.
    client._environment_opener = urllib.request.build_opener(NoRedirect())

    def request(url, method, path, payload, timeout):
        if url != endpoint:
            raise UnsupportedExecution("live Eval cannot change its SDK connection")
        return sdk_request(client, method, path, payload, timeout,
                           eval_submit=operation in {"run", "same_allocation_abba"} and method == "POST" and path == WIRE_PATH)
    return request


def sdk_request(client, method, path, payload, timeout, *, eval_submit=False):
    """Preserve SDK signing and errors; keyed auxiliary admission is one POST.

    The owner, not submit_job's hidden transport/overload loop, admits another
    auxiliary POST. Eval retains its existing SDK submission behavior.
    """
    from atrex_gateway_client.errors import GatewayError
    from .gateway.sandbox import GatewayHTTPError
    try:
        if eval_submit:
            client.timeout = timeout
            return client.submit_job("eval", payload)
        return client._request(method, path, payload, timeout)[1]
    except GatewayError as error:
        # Preserve the original SDK payload, including structured transport causes.
        original = error.payload
        sources = [original] if isinstance(original, dict) else []
        if sources and isinstance(original.get("error"), dict):
            sources.append(original["error"])
        sources.extend(value["details"] for value in tuple(sources) if isinstance(value.get("details"), dict))
        policies = {}
        for name in ("replay_policy", "execution_state", "retry_advice"):
            values = [value[name] for value in sources if name in value]
            if values:
                veto = "unknown" if name == "execution_state" else "never"
                policies[name] = veto if veto in values else values[0]
        native = {"category": error.error_class or "infrastructure", "code": error.reason or "sdk-http-error",
            "message": str(error), "details": {"original_sdk_error": {"status": error.status, "payload": original}},
            **policies}
        raise GatewayHTTPError(error.status, json.dumps(native)) from error
