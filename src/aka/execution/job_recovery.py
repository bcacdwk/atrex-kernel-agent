"""Selected backend recovery over confirmed job identities; no scheduler decisions."""
from aka.contracts.execution import UnsupportedExecution
from aka.contracts.execution_recovery import InfrastructureRetryPolicy, JobRecoveryCapabilities
from .gateway.jobs import GatewayJobs, selected_route


def recovery_policy(configuration):
    if type(configuration.get("idempotent_replay", False)) is not bool:
        raise ValueError("idempotent replay support must be an explicit boolean")
    return InfrastructureRetryPolicy(tuple(configuration.get("recovery_retry_delays", ())),
                                     configuration.get("recovery_retry_budget_s", 0))


class GatewayRecoveryMixin:
    def _job_transport(self):
        cached = getattr(self, "_selected_job_transport", None)
        if cached is not None:
            return cached
        configuration = getattr(self, "configuration", {})
        request = getattr(self, "request", None)
        environment = request.environment if request is not None else getattr(self, "environment", None)
        from .configuration import legacy_settings
        settings = request if request is not None else legacy_settings(environment)
        if settings.ssh:
            raise UnsupportedExecution("SSH executor does not provide durable job queries")
        endpoint = configuration.get("url") or settings.url
        profile = configuration.get("profile") or settings.profile
        if configuration.get("url"):
            profile = ""
        elif configuration.get("profile"):
            endpoint = ""
        from .gateway.sandbox import _find_agate
        executable = _find_agate()
        endpoint, profile, environment = selected_route(endpoint, profile, environment=environment,
                                                        executable=executable)
        supported = configuration.get("idempotent_replay", False)
        if supported:
            if not configuration.get("url"):
                raise UnsupportedExecution("stable-key replay needs a verified explicit gateway URL")
            executable = None
        elif endpoint:
            # Query the confirmed URL directly with frozen authentication. A later
            # CLI config change cannot redirect the job or introduce new credentials.
            executable = None
        transport = GatewayJobs(endpoint=endpoint, profile=profile, executable=executable,
                                environment=environment, idempotent_replay=bool(supported))
        self._selected_job_transport = transport
        return transport

    @property
    def recovery_capabilities(self):
        try:
            transport = self._job_transport()
        except UnsupportedExecution:
            return JobRecoveryCapabilities()
        query = bool(transport.endpoint or transport.executable)
        return JobRecoveryCapabilities(query, query, transport.idempotent_replay)

    def query_job(self, reference, *, timeout, retry=None):
        policy = recovery_policy(getattr(self, "configuration", {})) if retry is None else retry
        return self._job_transport().query(reference, timeout=timeout, retry=policy)

    def query_recorded_eval(self, association, *, timeout):
        """Read one retained live Eval through its original route and namespace."""
        from dataclasses import replace
        import math
        from urllib.parse import quote
        from .live_eval import digest, sdk_http, WIRE_PATH
        from .gateway.jobs import causes_from_error, client_wait_budget, result_from_job, unknown_result
        from aka.contracts.execution_recovery import JobReference
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("query timeout must be positive")
        transport = self._job_transport()
        reference = JobReference(**association["job_reference"])
        if (reference.backend != "gateway" or reference.operation != "run" or reference.profile
                or reference.endpoint != transport.endpoint or transport.profile
                or not reference.idempotency_key or not reference.request_identity
                or association.get("idempotency_key") != reference.idempotency_key
                or association.get("request_identity") != reference.request_identity
                or association.get("route_identity") != digest([reference.endpoint, WIRE_PATH, "", ""])
                or not transport.environment.get("AGATE_AK") or not transport.environment.get("AGATE_SK")
                or transport.environment.get("AGATE_TOKEN")
                or association.get("namespace_identity") != digest(transport.environment["AGATE_AK"])):
            raise UnsupportedExecution("recorded Eval recovery changed its original route or namespace")
        # Retain the live Eval's SDK signing and no-redirect connection. This
        # confirmation is one Job GET, with no retries or artifact collection.
        request = sdk_http(transport.environment, reference.endpoint, transport_retries=0)
        try:
            job = request(reference.endpoint, "GET", "/v1/jobs/" + quote(reference.identifier, safe="")
                          + "?include=spec", None, timeout)
            result = result_from_job(job, reference)
        except (OSError, ValueError, RuntimeError) as error:
            result = unknown_result(reference, type(error).__name__)
            causes = causes_from_error(error)
            result = replace(result, causes=causes, attempts=tuple(
                replace(item, causes=causes) for item in result.attempts))
        budgets, diagnostics = client_wait_budget("query-wait", timeout, timeout)
        return replace(result, budgets=(*result.budgets, *budgets),
                       diagnostics=(*result.diagnostics, *diagnostics))

    def cancel_job(self, reference, *, timeout):
        return self._job_transport().cancel(reference, timeout=timeout)

    def replay_job(self, command, *, idempotency_key):
        self.capabilities.require(command)
        from dataclasses import replace
        from .environment_query import command_query, require_supported
        from .gateway.requests import NATIVE_ABBA, bound_limits, prepared_operation
        if command.operation == NATIVE_ABBA:
            if command.submission is None:
                raise UnsupportedExecution("same-allocation ABBA requires an immutable prepared request")
            import json
            prepared_operation(json.loads(command.submission), command.operation)
        backend = self._environment_backend() if hasattr(self, "request") else self._environment_backend(command.environment)
        support = backend.check(command_query(command), discover=False)
        # Replay uses the same Agate typed protocol as first submission. Missing
        # discovery declarations remain unknown; explicit rejections still fail.
        require_supported(support, require_known=command.operation == NATIVE_ABBA and bool(backend.ssh))
        def decorate(result):
            return replace(result, environment_snapshot=support.environment,
                           environment_identity=support.environment.identity, environment_support=support)
        def observed(result):
            command.on_observation(decorate(result))
        bound = replace(command, environment=bound_limits(command.environment, support),
                        on_observation=observed if command.on_observation is not None else None)
        return decorate(self._job_transport().replay(bound, idempotency_key=idempotency_key,
            retry=recovery_policy(getattr(self, "configuration", {}))))
