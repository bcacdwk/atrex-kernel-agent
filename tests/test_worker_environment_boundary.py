"""CPU coverage of host/worker separation at the real native SDK boundary."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
import socket
import subprocess
from types import SimpleNamespace

import pytest

from aka.contracts.content import WorkspaceFile
from aka.contracts.evaluation import EvaluationConfiguration, EvaluationRequest, PreparationFile
from aka.contracts.execution import CommandSpec, ExecutionFile
from aka.contracts.problem import EvaluationInput
from aka.contracts.workspace import RevisionContent
from aka.evaluation.command_inputs import workload_environment
from aka.evaluation.evaluator import LegacyEvaluator
from aka.evaluation.gateway_command import GatewayCommandEvaluator
from aka.evaluation.tool_requests import request_from_argv
from aka.execution.legacy.executor import LegacyExecutor
from aka.workflow.schema.workflow import build_feedback
from test_schema_preparation_dispatch import native_launcher


PIN = "atrex-bench @ https://cpu.invalid/atrex_bench-0.1.0-py3-none-any.whl?Expires=1900000000&Signature=cpu-pin"
HOST = {
    "PATH": "/usr/bin:/bin", "HOME": "/cpu-private/host-home",
    "ANTHROPIC_API_KEY": "cpu-model-secret-sentinel",
    "OPENAI_API_KEY": "cpu-other-model-secret-sentinel",
    "AWS_SECRET_ACCESS_KEY": "cpu-host-secret-sentinel",
    "TRITON_CACHE_DIR": "/cpu-private/host-cache-sentinel",
    "BASH_FUNC_host%%": "() { echo cpu-shell-sentinel; }",
    "AGATE_AK": "cpu-fixed-ak", "AGATE_SK": "cpu-gateway-secret-sentinel",
    "AKA_GATEWAY_WARMUP_ITERS": "7",
    "ATREX_SANDBOX_ATREX_BENCH_VERSION": "cpu-pinned-build",
}
WORKER = {"TRITON_CACHE_DIR": "/tmp/t", "TORCH_LOGS": "+dynamo"}
SECRETS = tuple(HOST[key] for key in (
    "HOME", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY",
    "TRITON_CACHE_DIR", "BASH_FUNC_host%%", "AGATE_AK", "AGATE_SK"))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("worker environment CPU tests must not make network calls")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


def assert_private_absent(value):
    text = json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
    assert all(secret not in text for secret in SECRETS)
    assert "ANTHROPIC_API_KEY" not in text and "AWS_SECRET_ACCESS_KEY" not in text


@pytest.mark.parametrize("worker", [{}, WORKER])
def test_schema_v0_host_auth_stays_local_and_owner_matches_sdk_wire(tmp_path, monkeypatch, worker):
    # No SDK import skip: qualification requires the actual installed protocol.
    from atrex_gateway_client import Client, cli
    from atrex_gateway_client.auth import AkSkAuth
    from aka.execution import outcomes
    from aka.execution.gateway import requests, sandbox, submission

    campaign, driver, _, launches = native_launcher(tmp_path, monkeypatch,
        host_environment=HOST, requirements_text=PIN + "\n",
        evaluation_defaults=EvaluationConfiguration(worker_environment=worker))
    runtime = campaign._supervisor_runtime
    monkeypatch.setattr(LegacyExecutor, "_job_transport", lambda self: SimpleNamespace(
        environment=dict(HOST), endpoint="https://cpu.invalid", profile=""))
    client = Client("https://cpu.invalid", auth=AkSkAuth(HOST["AGATE_AK"], HOST["AGATE_SK"]),
                    timeout=1, transport_retries=0)
    original_run_owned = outcomes.run_owned
    posts, local_launches = [], []

    def retained():
        with runtime._record_reader() as reader:
            record, = reader.iter_records()
            artifact, = [item for item in record.artifacts if item.role == "submission_intent"]
            binding = json.loads(reader.read_artifact(artifact))
            source, = [item for item in record.artifacts if item.sha256 == binding["submission_sha256"]]
            raw = reader.read_artifact(source)
        assert hashlib.sha256(raw).hexdigest() == binding["submission_sha256"]
        assert binding["attempt_id"] == record.attempts[0].attempt_id
        assert_private_absent(raw.decode())
        return binding, json.loads(raw)

    def post(method, path, data, *args, **kwargs):
        assert (method, path) == ("POST", "/v1/jobs/eval")
        assert client.auth.access_key == HOST["AGATE_AK"]
        assert client.base_url == "https://cpu.invalid"
        binding, prepared = retained()
        payload = json.loads(data)
        expected = {**prepared["request"], "idempotency_key": binding["idempotency_key"]}
        expected["options"] = {**expected["options"], "timeout_s": binding["timeout"]}
        from aka.execution.live_eval import digest
        assert digest(payload) == digest(expected)
        assert payload["env_vars"] == worker
        assert payload["requirements"] == [PIN]
        assert payload["deps_mode"] == "freeze_installed"
        assert payload["spec"].get("num_gpus", 1) == 1
        assert set(payload["reference"]["shapes"]) == {"0", "1", "2"}
        assert payload["options"]["timeout_s"] == 60
        assert payload["runner_overrides"]["warmup_iters"] == 7
        assert_private_absent(payload)
        posts.append(payload)
        return 200, {"job_id": "cpu-original-job", "status": "queued", "kind": "eval"}

    def run_cli(argv, **kwargs):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr), pytest.raises(SystemExit) as stopped:
            cli.main(argv[1:])
        return subprocess.CompletedProcess(argv, stopped.value.code, stdout.getvalue(), stderr.getvalue())

    def inspect_launch(*args, **kwargs):
        # The actual bound launcher still has its host startup/auth environment.
        assert all(kwargs["environment"][key] == value for key, value in HOST.items() if key not in worker)
        assert runtime.environment["ANTHROPIC_API_KEY"] == HOST["ANTHROPIC_API_KEY"]
        binding, prepared = retained()
        payload = json.loads(kwargs["input_payload"])
        assert payload == {**prepared, "request": {
            **prepared["request"], "idempotency_key": binding["idempotency_key"]}}
        assert_private_absent(payload)
        local_launches.append(payload)
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
        assert len(posts) == len(local_launches) == len(launches) == 1
        assert driver.state.data["counts"]["dispatches"] == 1
        assert_private_absent(build_feedback(driver.state))
        record, = campaign.experiment_records()
        with runtime._record_reader() as reader:
            assert_private_absent(runtime.measurements.project(reader, record))
    finally:
        runtime.close()


def inputs(tmp_path, *, num_gpus=1, archive=False):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = {"kernel.py": b"class Model:\n    pass\n"}
    if archive:
        source["helper.py"] = b"# frozen auxiliary source\n"
    for name, data in source.items():
        (workspace / name).write_bytes(data)
    (workspace / "baseline.py").write_text("class Model:\n    pass\n# baseline\n")
    reference = tmp_path / "task"
    reference.mkdir()
    metadata = {} if num_gpus == 1 else {"hardware": {"required_gpus": num_gpus},
        "benchmark_contract": {"distributed_evaluation": {
            "launcher": "torchrun", "backend": "nccl", "world_size": num_gpus}}}
    metadata["shapes"] = {str(i): {"production_performance": {"performance_us": 10.0}} for i in range(3)}
    content = {"reference.py": "# trusted reference\n", "input.py": "# trusted inputs\n",
        "shapes.json": json.dumps({str(i): {"N": 32 * (i + 1)} for i in range(3)}),
        "metadata.json": json.dumps(metadata)}
    for name, text in content.items():
        (reference / name).write_text(text)
    task = EvaluationInput("legacy-native-v1", reference,
        tuple(WorkspaceFile(name, text=text) for name, text in content.items()), case_ids=("0", "1", "2"),
        worker_requirements=WorkspaceFile("requirements.txt", text=PIN + "\n"))
    provider = SimpleNamespace(read_source=lambda *args, **kwargs: RevisionContent(source),
        read_working=lambda path, names: RevisionContent({name: (path / name).read_bytes()
            if (path / name).is_file() else None for name in names}))
    return workspace, task, provider


@pytest.mark.parametrize("abba", [False, True])
@pytest.mark.parametrize("num_gpus", [1, 2])
def test_explicit_worker_override_retains_task_pin_shapes_and_repeats(tmp_path, abba, num_gpus):
    workspace, task, provider = inputs(tmp_path, num_gpus=num_gpus)
    config = EvaluationConfiguration(hardware="L20D", timeout=900, repeats=4, per_run_timeout=73,
        environment=HOST, worker_environment=WORKER, shape_batch_size=2, measurement_repetitions=3)
    argv = ["--kind", "run", "--env", "TORCH_LOGS=+inductor"]
    if abba:
        argv += ["--baseline-path", "baseline.py"]
    command = request_from_argv(argv, workspace=workspace, evaluation_input=task,
        configuration=config, workspace_provider=provider).request
    plan = GatewayCommandEvaluator(execution_capabilities=LegacyExecutor.capabilities,
        evaluation_input=task, shape_batch_size=2, measurement_repetitions=3).prepare(command)
    assert len(plan.steps) == 6
    for step in plan.steps:
        payload = json.loads(step.submission)["request"]
        assert payload["env_vars"] == {**WORKER, "TORCH_LOGS": "+inductor"}
        assert step.requirements == (PIN,) and payload["requirements"] == [PIN]
        assert payload["spec"].get("num_gpus", 1) == step.num_gpus == num_gpus
        assert payload["options"]["timeout_s"] == step.timeout == 900
        assert payload["candidate"] == (workspace / "kernel.py").read_text()
        assert payload["reference"]["reference_py"] == "# trusted reference\n"
        assert payload["runner_overrides"]["warmup_iters"] == 7
        if abba:
            assert payload["abba"] == {"baseline": (workspace / "baseline.py").read_text(), "repeats": 2}
            assert payload["runner_overrides"]["perf_timeout_s"] == 73
        else:
            assert "abba" not in payload
        assert_private_absent(payload)
    for start in range(0, len(plan.steps), 2):
        assert {shape for step in plan.steps[start:start + 2]
                for shape in json.loads(step.submission)["request"]["reference"]["shapes"]} == {"0", "1", "2"}


@pytest.mark.parametrize("mode", ["v0", "abba"])
@pytest.mark.parametrize("worker", [{}, WORKER])
def test_legacy_typed_and_native_abba_share_the_environment_boundary(tmp_path, mode, worker):
    workspace, task, provider = inputs(tmp_path)
    config = EvaluationConfiguration(hardware="L20D", timeout=900, repeats=4, per_run_timeout=73,
        environment=HOST, worker_environment=worker, shape_batch_size=2, measurement_repetitions=3,
        comparison_method="native-abba", execution_capabilities=LegacyExecutor.capabilities)
    plan = LegacyEvaluator(config, candidate_workspace=provider, evaluation_input=task).prepare(
        EvaluationRequest(workspace, mode=mode, base_commit="base", candidate_commit="candidate",
                          changed_paths=("kernel.py",), expected_shape_ids=frozenset(task.case_ids)))
    assert len(plan.steps) == 6
    for step in plan.steps:
        payload = json.loads(step.submission)["request"]
        assert payload["env_vars"] == worker
        assert workload_environment(step) == worker
        assert payload["requirements"] == [PIN]
        assert payload["options"]["timeout_s"] == 900
        assert payload["runner_overrides"]["warmup_iters"] == 7
        if mode == "abba":
            assert payload["abba"]["repeats"] == 2
            assert payload["runner_overrides"]["perf_timeout_s"] == 73
        assert_private_absent(payload)


@pytest.mark.parametrize("item", ["AGATE_AK=wrong-namespace", "ATREX_SANDBOX_URL=https://wrong.invalid", "bad-key=value", "=value", "missing-equals"])
def test_invalid_explicit_tool_values_are_rejected_not_dropped(tmp_path, item):
    workspace, task, provider = inputs(tmp_path)
    with pytest.raises(ValueError):
        request_from_argv(["--env", item], workspace=workspace, evaluation_input=task,
            configuration=EvaluationConfiguration(hardware="L20D", environment=HOST), workspace_provider=provider)


@pytest.mark.parametrize("worker", [{"AGATE_SK": "invalid"}, {"bad-key": "invalid"}, {"TORCH_LOGS": 1}])
def test_invalid_explicit_configuration_values_are_rejected(tmp_path, worker):
    workspace, task, provider = inputs(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        request_from_argv([], workspace=workspace, evaluation_input=task,
            configuration=EvaluationConfiguration(hardware="L20D", worker_environment=worker), workspace_provider=provider)


def test_unknown_explicit_worker_key_reaches_service_validation_unchanged(tmp_path):
    workspace, task, provider = inputs(tmp_path)
    command = request_from_argv(["--env", "UNLISTED_WORKER_KEY=caller-request"], workspace=workspace,
        evaluation_input=task, configuration=EvaluationConfiguration(hardware="L20D", environment=HOST),
        workspace_provider=provider).request
    assert json.loads(command.submission)["request"]["env_vars"] == {"UNLISTED_WORKER_KEY": "caller-request"}


@pytest.mark.parametrize("ssh", [False, True])
def test_archive_and_ssh_typed_envelopes_keep_only_explicit_worker_values(tmp_path, monkeypatch, ssh):
    from aka.execution.gateway.source_archives import prepared_payload, read_archives
    from aka.evaluation.source_archives import archive_source
    monkeypatch.setattr("aka.execution.configuration.selected_ssh", lambda: ssh)
    workspace, task, provider = inputs(tmp_path, archive=True)
    command = request_from_argv(["--env", "TORCH_LOGS=+dynamo"], workspace=workspace,
        evaluation_input=task, configuration=EvaluationConfiguration(hardware="L20D", environment=HOST),
        workspace_provider=provider).request
    payload = json.loads(prepared_payload(replace(command, idempotency_key="cpu-archive-key"), ssh=ssh))
    assert payload["request"]["env_vars"] == {"TORCH_LOGS": "+dynamo"}
    archive, = read_archives(payload)
    assert archive_source(archive).files == provider.read_source(workspace).files
    assert payload["request"]["requirements"] == [PIN]
    assert_private_absent(payload)


def test_public_command_and_retained_cli_explicit_environments_still_work(tmp_path):
    from aka.evaluation.cli_adapter import prepare_command
    command = CommandSpec(tmp_path, ("python3", "test_kernel.py"), environment=WORKER)
    assert workload_environment(command) == WORKER
    cli_command = prepare_command(replace(command, environment=HOST), environment_items=("TORCH_LOGS=+dynamo",))
    assert workload_environment(cli_command) == {"TORCH_LOGS": "+dynamo"}
    assert cli_command.environment["AGATE_AK"] == HOST["AGATE_AK"]


def test_explicit_configuration_is_frozen_before_preparation():
    worker = dict(WORKER)
    config = EvaluationConfiguration(worker_environment=worker)
    worker["TORCH_LOGS"] = "changed-after-configuration"
    assert dict(config.worker_environment) == WORKER


@pytest.mark.parametrize("worker", [{}, WORKER])
def test_typed_supplemental_probe_freezes_the_same_explicit_worker_values(tmp_path, worker):
    from aka.evaluation.typed_numerical import prepare_typed
    from aka.execution.gateway.executor import GatewayExecutor
    workspace, task, _ = inputs(tmp_path)
    candidate = (PreparationFile("kernel.py", (workspace / "kernel.py").read_bytes()),
                 PreparationFile("solution.json", b'{"languages": ["triton"]}'))
    request = SimpleNamespace(workspace=workspace, numerical_probe=SimpleNamespace(
        candidate_files=candidate, rotation="cpu-probe", candidate_identity="0" * 64))
    config = EvaluationConfiguration(hardware="L20D", environment=HOST, worker_environment=worker,
        execution_capabilities=GatewayExecutor.capabilities)
    trusted = {item.path: ExecutionFile(item.path, item.text.encode()) for item in task.files}
    plan = prepare_typed(request, config, {"world_size": 1, "cases": [{"id": "edge"}]},
        {"0": {"N": 32}}, [{"case_id": "edge", "shape_ids": ["0"], "seeds": [1],
                            "selection_digest": "cpu-selection"}], trusted, 0.1, (PIN,))
    step, = plan.steps
    payload = json.loads(step.submission)["request"]
    assert workload_environment(step) == payload["env_vars"] == worker
    assert payload["mode"] == "correctness_only"
    assert payload["requirements"] == [PIN]
    assert payload["options"]["num_correctness_cases"] == 1
    assert payload["runner_overrides"]["warmup_iters"] == 7
    assert_private_absent(payload)
