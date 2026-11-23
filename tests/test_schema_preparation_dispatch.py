"""Native preparation, Experiment and launcher wiring with a CPU child process."""
from contextlib import ExitStack
from contextvars import copy_context
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import pytest

from aka.contracts.content import WorkspaceFile
from aka.contracts.evaluation import EvaluationConfiguration
from aka.contracts.execution import CommandSpec, ExecutionContext
from aka.contracts.execution_environment import EnvironmentQuery
from aka.contracts.experiment_identity import ExperimentRef
from aka.contracts.problem import EvaluationInput
from aka.evaluation.plugin import LegacyEvaluatorFactory
from aka.execution.environment_query import _snapshot, match_environment
from aka.execution.legacy.executor import LegacyExecutor
from aka.experiment.factory import LegacyExperimentFactory
from aka.workflow.light.orchestration.campaign import Campaign
from aka.workflow.light.runtime import ManagedRuntimeFactory
from aka.workflow.schema.run_state import BudgetDenied, RecoveryRequired, RunLimits
from aka.workflow.schema.workflow import _measurement_feedback, build_feedback
from test_schema_workflow import controller, native


def native_launcher(tmp_path, monkeypatch, *, host_environment=None,
                    evaluation_defaults=None, requirements_text=""):
    campaign = native(tmp_path)
    task = tmp_path / "task-inputs"
    task.mkdir()
    example = Path(__file__).resolve().parents[1] / "examples/schema/l20d/vector_add/benchmark"
    files = tuple(WorkspaceFile(name, text=(example / name).read_text())
                  for name in ("reference.py", "input.py", "shapes.json", "metadata.json"))
    requirements = task / "requirements.txt"
    requirements.write_text(requirements_text)
    evaluation = EvaluationInput("legacy-native-v1", task, files,
        runtime_root=task, case_ids=("0", "1", "2"),
        worker_requirements=WorkspaceFile("requirements.txt", source=requirements))
    campaign.problem.evaluation = evaluation
    original_content = campaign.problem_provider.content
    def content(*args, **kwargs):
        value = original_content(*args, **kwargs)
        value.evaluation = evaluation
        return value
    campaign.problem_provider.content = content
    executor = LegacyExecutor()
    workflow = SimpleNamespace(executor=executor, agent=campaign.agent,
        problem=campaign.problem_provider, execution_context=ExecutionContext(host_environment or {}, "L20D", 60),
        evaluation_defaults=evaluation_defaults or EvaluationConfiguration(), measurement_parallelism=1,
        evaluator_factory=LegacyEvaluatorFactory(), experiment_factory=LegacyExperimentFactory(),
        options=SimpleNamespace(wiki=False, research_records=False, wiki_options={},
            request_timeout=60, queue_timeout=60, inherit_state=False))
    campaign.tool_runtime = ManagedRuntimeFactory(workflow, campaign)
    campaign.sandbox_timeout = 60
    campaign._supervisor_runtime = None
    campaign.start_runtime = MethodType(Campaign.start_runtime, campaign)
    campaign.measure_for_acceptance = MethodType(Campaign.measure_for_acceptance, campaign)

    fields = json.loads((Path(__file__).parent / "fixtures/agate_b41_environment.json").read_text())
    environment = _snapshot("legacy", "cpu-fixture", EnvironmentQuery("L20D"), fields["detail"],
        fields["capabilities"], [], LegacyExecutor.capabilities.operations)
    support = match_environment(EnvironmentQuery("L20D", "run"), environment)
    monkeypatch.setattr(LegacyExecutor, "_environment_backend",
        lambda self: SimpleNamespace(backend="legacy", endpoint="cpu-fixture", check=lambda *a, **k: support))
    monkeypatch.setattr(LegacyExecutor, "_job_transport",
        lambda self: SimpleNamespace(environment={}, endpoint="", profile=""))

    # Keep run_owned, its observation pipe, Experiment persistence and native
    # evaluation real. Substitute only the final launcher process's program.
    from aka.execution.processes import spawn_owned_session
    launches = []
    result = {"job_id": "synthetic-cpu-v0", "status": "succeeded", "result": {
        "passed": {"compile": {"status": "passed"},
            "correctness": {sid: {"status": "passed"} for sid in evaluation.case_ids}},
        "performance": {"shapes": {sid: {"samples": [{"end_to_end_time_ms": .01}]}
            for sid in evaluation.case_ids}},
        "private_payload": "DO_NOT_PROJECT_RAW_NATIVE_PAYLOAD"}}
    script = "\n".join(("import json, os",
        "submitted = {'event': 'submitted', 'attempt_id': 'cpu-attempt', "
        "'operation': 'run', 'backend': 'gateway', 'identifier': 'synthetic-cpu-v0'}",
        "os.write(int(os.environ['AKA_EXECUTION_FACTS_FD']), (json.dumps(submitted) + '\\n').encode())",
        "fact = {'event': 'attempt', 'attempt_id': 'cpu-attempt', 'operation': 'run', "
        "'backend': 'gateway', 'identifier': 'synthetic-cpu-v0', 'disposition': 'terminal', 'termination': 'exited', "
        "'exit_code': 0, 'remote_terminated': True}",
        "os.write(int(os.environ['AKA_EXECUTION_FACTS_FD']), (json.dumps(fact) + '\\n').encode())",
        "print(" + repr(json.dumps(result)) + ")"))
    def spawn(argv, **kwargs):
        launches.append(tuple(argv))
        return spawn_owned_session([sys.executable, "-c", script], **kwargs)
    monkeypatch.setattr("aka.execution.processes.spawn_owned_session", spawn)

    # This is the real early binding that the old copy-at-bind code missed.
    campaign.start_runtime()
    runtime = campaign._supervisor_runtime
    assert runtime.measurements.executor is runtime.executor
    assert runtime.executor.executor is not executor
    assert runtime.executor.executor._dispatch_scope is None
    driver, planner = controller(tmp_path, campaign,
        limits=RunLimits(max_episodes=1, max_dispatches=1))
    driver.executor = driver._recovery_executor = executor
    return campaign, driver, planner, launches


def test_native_v0_prebound_runtime_is_admitted_once_and_feedback_is_bound(tmp_path, monkeypatch):
    campaign, driver, planner, launches = native_launcher(tmp_path, monkeypatch)
    runtime = campaign._supervisor_runtime
    observed_owner = []
    original_observe = driver.state.observe_execution
    def observe(permit, **event):
        # The native owner persists its result before the Schema observer.
        records = campaign.experiment_records()
        observed_owner.append(any(attempt.result is not None for record in records for attempt in record.attempts))
        original_observe(permit, **event)
    driver.state.observe_execution = observe
    try:
        with driver.state.owned(), ExitStack() as stack:
            driver._scopes(stack, campaign)
            driver._prepare(campaign)
            state = driver.state.data
            assert len(launches) == state["counts"]["dispatches"] == 1
            assert all(call["state"] == "settled" for call in state["calls"].values())
            record, = campaign.experiment_records()
            assert record.kind == "command" and record.evaluation.measurements == ()
            summary, = json.loads(build_feedback(driver.state))["preparation_measurements"]
            assert summary["experiment"] == record.ref.as_dict()
            assert summary["candidate_source_identities"] == [state["champion"]["identity"]]
            assert summary["prepared_source_match"] and summary["prepared_id"] == state["prepared"]["identity"]
            assert summary["candidate_latency_us"] == pytest.approx(10.) and summary["gate"] == "PASS"
            assessment, = summary["measurements"]
            assert assessment["completeness"] == "complete" and assessment["validity"] == "valid"
            assert assessment["comparability"] == record.attempts[0].assessment.comparability
            assert assessment["environment_identity"] == record.attempts[0].assessment.environment_identity
            assert "DO_NOT_PROJECT" not in build_feedback(driver.state)
            assert observed_owner and all(observed_owner)

            # A bound copy made after attachment and a copied context use the
            # same before-launch allowance. There is no nested/double charge.
            late = driver.executor.bind(ExecutionContext({}, "L20D", 60))
            command = CommandSpec(campaign.workspace, ("true",), hardware="L20D", timeout=60, operation="dev")
            with pytest.raises(BudgetDenied, match="dispatches"):
                copy_context().run(late.run, command)
            # Change source so execute_trusted cannot reuse V0. The same native
            # path must be denied before launching a second physical process.
            with (campaign.workspace / "kernel.py").open("a") as output:
                output.write("\n# new candidate must obtain its own dispatch\n")
            with pytest.raises(BudgetDenied, match="dispatches"):
                campaign.measure_for_acceptance(campaign.workspace, ["--kind", "run", "--mode", "full", "--no-sync"])
            assert len(launches) == state["counts"]["dispatches"] == 1
        for stale in (runtime.executor.executor, late):
            with pytest.raises(RuntimeError, match="closed"):
                stale.run(command)
            with pytest.raises(RuntimeError, match="closed"):
                stale.bind(ExecutionContext({}, "L20D", 60)).run(command)
    finally:
        runtime.close()


def test_native_v0_retained_key_reaches_launcher_and_sdk(tmp_path, monkeypatch):
    from contextlib import redirect_stderr, redirect_stdout
    import hashlib
    import io
    import subprocess
    from atrex_gateway_client import Client, cli
    from aka.execution import outcomes
    from aka.execution.gateway import requests, sandbox, submission

    campaign, driver, _, launches = native_launcher(tmp_path, monkeypatch)
    runtime = campaign._supervisor_runtime
    original_run_owned = outcomes.run_owned
    client = Client("https://cpu.invalid", timeout=1, transport_retries=0)
    inspected, wire_keys = [], []

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def retained():
        # Reopen the owner's records at each boundary, before any fake POST.
        with runtime._record_reader() as reader:
            record, = reader.iter_records()
            artifact, = [item for item in record.artifacts if item.role == "submission_intent"]
            binding = json.loads(reader.read_artifact(artifact))
            original, = [item for item in record.artifacts if item.sha256 == binding["submission_sha256"]]
            raw = reader.read_artifact(original)
        assert hashlib.sha256(raw).hexdigest() == binding["submission_sha256"]
        assert binding["attempt_id"] == record.attempts[0].attempt_id
        assert binding["operation"] == "run"
        assert binding["idempotency_key"].startswith("experiment:")
        return binding, json.loads(raw)

    def post(method, path, data, *args, **kwargs):
        assert (method, path) == ("POST", "/v1/jobs/eval")
        binding, prepared = retained()
        payload = json.loads(data)
        assert payload["idempotency_key"] == binding["idempotency_key"]
        expected = {**prepared["request"], "idempotency_key": binding["idempotency_key"]}
        expected["options"] = {**expected["options"], "timeout_s": binding["timeout"]}
        assert digest(payload) == digest(expected)
        wire_keys.append(payload["idempotency_key"])
        return 200, {"job_id": "cpu-original-job", "status": "queued", "kind": "eval"}

    def run_cli(argv, **kwargs):
        assert argv[1] == "submit" and argv[-1] == "--no-wait"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr), pytest.raises(SystemExit) as stopped:
            cli.main(argv[1:])
        return subprocess.CompletedProcess(argv, stopped.value.code, stdout.getvalue(), stderr.getvalue())

    def inspect_launch(*args, **kwargs):
        binding, prepared = retained()
        payload = json.loads(kwargs["input_payload"])
        key = payload["request"].get("idempotency_key")
        assert key == binding["idempotency_key"], {"retained_key": binding["idempotency_key"], "launcher_key": key}
        prepared["request"]["idempotency_key"] = key
        assert digest(payload) == digest(prepared)
        inspected.append(key)
        # Exercise the actual submission/SDK adapter with these exact launch
        # bytes. Only HTTP and the external CLI process are CPU substitutes.
        with monkeypatch.context() as transport:
            transport.setattr(cli, "_client", lambda args: client)
            transport.setattr(cli._update, "maybe_warn_outdated", lambda version: None)
            transport.setattr(client, "_request_once", post)
            transport.setattr(sandbox, "_find_agate", lambda: "/cpu/agate")
            transport.setattr(sandbox, "_submission_route", lambda *args: ("https://cpu.invalid", ""))
            transport.setattr(sandbox.subprocess, "run", run_cli)
            transport.setattr(sandbox, "_resume_interrupted_agate_wait", lambda **kwargs:
                subprocess.CompletedProcess((), 0, json.dumps({"job_id": "cpu-original-job",
                    "status": "succeeded", "result": {}}), ""))
            for module in (outcomes, sandbox, submission, requests):
                transport.setattr(module, "emit_fact", lambda value: None)
            options = SimpleNamespace(kind="run", url="https://cpu.invalid", gateway_profile=None,
                ssh=False, input=[], timeout=binding["timeout"])
            assert submission.submit_prepared(options, kwargs["workspace"], 0, payload) == 0
        return original_run_owned(*args, **kwargs)

    monkeypatch.setattr(outcomes, "run_owned", inspect_launch)
    try:
        with driver.state.owned(), ExitStack() as stack:
            driver._scopes(stack, campaign)
            driver._prepare(campaign)
        assert inspected == wire_keys
        assert len(inspected) == len(launches) == driver.state.data["counts"]["dispatches"] == 1
    finally:
        runtime.close()


def test_scope_owner_reopens_without_reviving_old_bindings(tmp_path, monkeypatch):
    campaign, driver, _, launches = native_launcher(tmp_path, monkeypatch)
    executor = driver.executor
    old = campaign._supervisor_runtime.executor.executor
    command = CommandSpec(tmp_path, ("true",), hardware="L20D", timeout=60, operation="dev")
    events = []
    try:
        # Ordinary no-scope behavior remains usable, including early bindings.
        assert old.run(command).returncode == 0
        for _ in range(2):
            with executor.dispatch_scope(admit=lambda **kw: events.append("admit") or "p",
                    observe=lambda permit, **kw: events.append(kw["event"])):
                fresh = executor.bind(ExecutionContext({}, "L20D", 60))
                with pytest.raises(RuntimeError, match="already belongs"):
                    with fresh.dispatch_scope(admit=lambda **kw: None, observe=lambda *a, **kw: None):
                        pass
                assert copy_context().run(fresh.run, command).returncode == 0
            with pytest.raises(RuntimeError, match="closed"):
                old.run(command)
        assert events.count("admit") == 2 and len(launches) == 3
        assert executor.run(command).returncode == 0
        assert len(launches) == 4
    finally:
        campaign._supervisor_runtime.close()


@pytest.mark.parametrize("broken", ["identity", "unknown", "latest"])
def test_preparation_feedback_rejects_unmatched_or_unsettled_attempts(tmp_path, monkeypatch, broken):
    campaign, driver, _, _ = native_launcher(tmp_path, monkeypatch)
    try:
        with driver.state.owned(), ExitStack() as stack:
            driver._scopes(stack, campaign)
            driver._prepare(campaign)
        record, = campaign.experiment_records()
        attempt, = record.attempts
        if broken == "identity":
            attempt = replace(attempt, assessment=replace(attempt.assessment,
                identity=replace(attempt.assessment.identity, input_digest="f" * 64)))
        elif broken == "unknown":
            attempt = replace(attempt, result=replace(attempt.result,
                facts=replace(attempt.result.facts, disposition="unknown")))
        else:
            attempt = replace(attempt, attempt_id="later", disposition="not_submitted", result=None, assessment=None)
        changed = replace(record, attempts=(*record.attempts, attempt) if broken == "latest" else (attempt,))
        assert _measurement_feedback(changed)["measurements"] == []
    finally:
        campaign._supervisor_runtime.close()


def test_repaired_resume_refreshes_only_original_preparation_without_refund_or_dispatch(tmp_path, monkeypatch):
    campaign, driver, _, launches = native_launcher(tmp_path, monkeypatch)
    runtime = campaign._supervisor_runtime
    try:
        with driver.state.owned(), ExitStack() as stack:
            driver._scopes(stack, campaign)
            driver._prepare(campaign)
            data = driver.state.data
            original_seed = data["champion"]
            record, = campaign.experiment_records()
            # Reproduce an older root's sparse persisted public projection.
            data["prepared"].pop("source_identity")
            data["preparation_measurements"] = [{"experiment": record.ref.as_dict(), "measurements": []}]
            operation_id = driver.state.action("optimize")
            data["phase"] = "optimize"
            frozen = {"operation_id": operation_id, "feedback": [{"text": "original frozen inputs"}]}
            data["planner_request"] = frozen
            permit = driver.state.admit_agent(command=("cpu-model",), workspace=tmp_path,
                timeout=60, environment={}, backend="claude", access=None, control=None)
            driver.state.observe_agent(permit, event="error", value=KeyboardInterrupt())
            data["calls"][permit]["charged_seconds"] = 3.
            driver.state.checkpoint_active_time(3., 0)
        with pytest.raises(RecoveryRequired, match="unresolved"):
            driver.resume_after_repair("requires the original model owner's terminal evidence")
        assert len(launches) == 1 and data["calls"][permit]["state"] == "unknown"
        driver.recover_agent = lambda key, owner: {"terminal": True, "active_seconds": 3.,
            "basis": "synthetic original local owner exited"} if key == permit else None
        driver.resume_after_repair("original owner confirmed stopped; preparation feedback repair")
        data = driver.state.data
        assert data["calls"][permit]["state"] == "settled" and data["counts"]["agent_calls"] == 1
        assert data["counts"]["agent_active_seconds"] == 3.
        counts = dict(data["counts"])

        # Later champion/Experiment facts must not be relabeled as initial V0.
        other_ref = ExperimentRef(record.ref.run_id, "unrelated-later-experiment")
        unrelated = replace(record, ref=other_ref,
            artifacts=tuple(replace(item, experiment=other_ref) for item in record.artifacts),
            evaluation=replace(record.evaluation, experiment_ref=other_ref, candidate_latency_us=1.))
        monkeypatch.setattr(runtime, "read_experiment_records", lambda: (record, unrelated))
        with driver.state.owned():
            data = driver.state.data
            data["hypotheses"]["initial"] = driver._new_hypothesis(original_seed)
            data["champion"] = {**original_seed, "identity": "later-champion", "origin": {"role": "champion"}}
            data["working_draft"] = data["champion"]
            driver._prepare(campaign)
        summary, = data["preparation_measurements"]
        assert summary["experiment"] == record.ref.as_dict()
        assert summary["candidate_source_identities"] == [original_seed["identity"]]
        assert summary["prepared_source_match"] and summary["candidate_latency_us"] == pytest.approx(10.)
        assert summary["measurements"][0]["validity"] == "valid"
        assert data["counts"] == counts and len(launches) == 1
        assert data["planner_request"] == frozen and data["action"] is None
        mismatch, = driver._preparation_feedback(campaign, (record,),
            prepared_id=data["prepared"]["identity"], source_identity="different-prepared-source")
        assert not mismatch["prepared_source_match"]
        assert mismatch["candidate_source_identities"] == [original_seed["identity"]]
        monkeypatch.setattr(runtime, "read_experiment_records", lambda: (unrelated,))
        with driver.state.owned(), pytest.raises(RecoveryRequired, match="preparation Experiment"):
            driver._prepare(campaign)
    finally:
        runtime.close()


@pytest.mark.parametrize("complete_evidence", [True, False])
def test_historical_unadmitted_v0_consumption_is_explicit_exact_and_idempotent(tmp_path, monkeypatch, complete_evidence):
    campaign, driver, _, launches = native_launcher(tmp_path, monkeypatch)
    runtime = campaign._supervisor_runtime
    # Reproduce the old pre-binding defect: this native service's executor has
    # copied configuration, but no shared attachment to the later root scope.
    early = runtime.executor.executor
    runtime.executor.executor = LegacyExecutor(early.request, early.hooks, early.configuration)
    try:
        with driver.state.owned(), ExitStack() as stack:
            driver._scopes(stack, campaign)
            driver._prepare(campaign)
        assert len(launches) == 1 and driver.state.data["counts"]["dispatches"] == 0
        with driver.state.owned(), pytest.raises(RecoveryRequired, match="historical usage repair"):
            driver._prepare(campaign)
        record, = campaign.experiment_records()
        if not complete_evidence:
            attempt, = record.attempts
            incomplete = replace(record, attempts=(replace(attempt,
                result=replace(attempt.result, job_references=())),))
            monkeypatch.setattr(runtime, "read_experiment_records", lambda: (incomplete,))
            with pytest.raises(RecoveryRequired, match="count is unknown"):
                driver.resume_after_repair("Cannot establish original dispatch count", preparation_campaign=campaign)
            assert driver.state.data["counts"]["dispatches"] == 0
            assert not driver.state.data.get("preparation_dispatch_corrections")
            assert len(launches) == 1
            return
        for _ in range(2):
            driver.resume_after_repair("Legacy pre-binding bypassed preparation admission",
                preparation_campaign=campaign)
            assert driver.state.data["counts"]["dispatches"] == 1
            assert driver.state.data["calls"] == {}  # No invented past admission.
        correction, = driver.state.data["preparation_dispatch_corrections"].values()
        assert correction["kind"] == "historical-unadmitted"
        assert correction["previous_dispatches"] == 0 and correction["consumed_dispatches"] == 1
        assert correction["evidence"]["experiment"] == record.ref.as_dict()
        assert correction["evidence"]["attempt_id"] == record.attempts[0].attempt_id
        with driver.state.owned(), ExitStack() as stack:
            driver._scopes(stack, campaign)
            driver._prepare(campaign)
            driver.state.action("optimize")
            command = CommandSpec(tmp_path, ("true",), hardware="L20D", timeout=60, operation="dev")
            with pytest.raises(BudgetDenied, match="dispatches"):
                driver.executor.bind(ExecutionContext({}, "L20D", 60)).run(command)
        assert len(launches) == 1 and driver.state.data["counts"]["dispatches"] == 1
    finally:
        runtime.close()
