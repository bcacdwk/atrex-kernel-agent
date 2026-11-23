"""Public fresh GEMM V0 through the real Sandbox, recorded owner and native SDK."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from aka.contracts.execution_environment import EnvironmentQuery
from aka.execution.environment_query import _snapshot, match_environment
from aka.execution.legacy.executor import LegacyExecutor
from aka.execution.live_eval import digest


CHILD = r'''
import hashlib, json, os, sys
from pathlib import Path
assert sys.dont_write_bytecode and os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
def guard(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise AssertionError("CPU fixture forbids network")
sys.addaudithook(guard)
from atrex_gateway_client import Client, cli
from atrex_gateway_client.auth import AkSkAuth
from aka.execution.gateway import sandbox
state_path = Path(os.environ["CPU_STATE"])
frozen = json.loads(Path(os.environ["CPU_EFFECTIVE_COMMAND"]).read_bytes())
expected = frozen["environment"]
# Legacy adds its launcher protocol after receiving the effective CommandSpec.
# Nonempty Task requirements have always supplied this derived launcher value;
# the prepared worker request and full CommandSpec identity remain independent.
launcher = dict(expected)
if frozen["requirements"]:
    launcher["AKA_GATEWAY_DEV_DEPENDENCIES"] = json.dumps(frozen["requirements"])
assert all(os.environ.get(key) == value for key, value in launcher.items())
from aka.execution.live_eval import digest
command_digest = digest(expected)
launcher_digest = digest({key: os.environ[key] for key in launcher})
def request(self, method, path, data, *args, **kwargs):
    state = json.loads(state_path.read_text())
    payload = json.loads(data) if data else None
    state["command_environment_sha256"] = command_digest
    state["launcher_environment_sha256"] = launcher_digest
    state["calls"].append({"method": method, "path": path, "payload": payload})
    if method == "POST":
        assert path == "/v1/jobs/eval"
        state["body"] = payload
        state_path.write_text(json.dumps(state))
        return 200, {"job_id": "cpu-effective-v0", "status": "queued", "kind": "eval"}
    assert path.startswith("/v1/jobs/cpu-effective-v0")
    body = state["body"]
    state_path.write_text(json.dumps(state))
    shapes = body["reference"]["shapes"]
    return 200, {"job_id": "cpu-effective-v0", "status": "succeeded", "num_gpus": 1,
        "request_digest": {"gpu": "NVIDIA L20D", "num_gpus": 1}, "result": {
        "passed": {"compile": {"status": "passed"},
                   "correctness": {sid: {"status": "passed"} for sid in shapes}},
        "performance": {"shapes": {sid: {"samples": [{"end_to_end_time_ms": .01}]} for sid in shapes}},
        "dsl": "triton", "environment": {"gpu_name": "NVIDIA L20D", "gpu_arch": "sm_103", "triton_version": "3.7.1"},
        "shape_selection": {"all_shape_ids": list(shapes), "partial": False}}}
Client._request_once = request
sandbox._find_agate = lambda: "/cpu/agate"
cli._client = lambda args: Client(args.url, auth=AkSkAuth("cpu-ak", "cpu-sk"), timeout=1, transport_retries=0)
cli._update.maybe_warn_outdated = lambda version: None
import contextlib, io, subprocess
def cpu_cli(argv, **kwargs):
    assert argv[0] == "/cpu/agate"
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            cli.main(argv[1:])
        except SystemExit as stopped:
            code = stopped.code
    return subprocess.CompletedProcess(argv, code, stdout.getvalue(), stderr.getvalue())
sandbox.subprocess.run = cpu_cli
raise SystemExit(sandbox.main())
'''


def native_cpu(tmp_path, monkeypatch, *, requirements_text="triton==3.7.1\n"):
    import atrex_gateway_client
    from aka.execution.processes import spawn_owned_session
    root = Path(__file__).resolve().parents[1]
    task = tmp_path / "native/problems/fp8_scale_group_gemm"
    task.mkdir(parents=True)
    example = root / "examples/schema/l20d/fp8_scale_group_gemm/benchmark"
    for source in example.iterdir():
        (task / source.name).write_bytes(source.read_bytes())
    (tmp_path / "native/scripts").mkdir()
    (tmp_path / "native/scripts/run_eval.py").write_text("raise AssertionError('CPU must use prepared SDK')\n")
    (tmp_path / "native/src/atrex_bench").mkdir(parents=True)
    (tmp_path / "native/src/atrex_bench/__init__.py").write_text("")
    requirements = tmp_path / "worker.txt"
    requirements.write_text(requirements_text)
    child = tmp_path / "native_child.py"
    child.write_text(CHILD)
    service = tmp_path / "service.json"
    service.write_text(json.dumps({"calls": []}))
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "claude").write_text("#!/bin/sh\nexit 99\n")
    (tools / "claude").chmod(0o700)
    (tools / "agate").write_text("#!/bin/sh\nexit 99\n")
    (tools / "agate").chmod(0o700)
    environment = {"PATH": str(tools) + ":/usr/bin:/bin", "HOME": str(tmp_path),
        "PYTHONDONTWRITEBYTECODE": "1", "CPU_STATE": str(service),
        "CPU_EFFECTIVE_COMMAND": str(tmp_path / "effective-command.private.json"),
        "PYTHONPATH": os.pathsep.join((str(root / "src"), str(Path(atrex_gateway_client.__file__).resolve().parents[1]))),
        "AGATE_AK": "cpu-ak", "AGATE_SK": "cpu-sk", "ATREX_SANDBOX_QUEUE_WAIT_GRACE": "0",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "cpu-model", "ANTHROPIC_DEFAULT_OPUS_MODEL": "cpu-model",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "cpu-model", "ATREX_CLAUDE_SESSION_SETTINGS": "{}",
        "ATREX_ENVIRONMENT_RESTART_SUPERVISED": "1", "CLAUDE_CODE_DISABLE_AUTOUPDATER": "1",
        "DISABLE_AUTOUPDATER": "1", "PYTHONUNBUFFERED": "1", "ATREX_AGATE_EXECUTABLE": str(tools / "agate"),
        "ANTHROPIC_MODEL": "cpu-model", "ANTHROPIC_BASE_URL": "https://cpu-model.invalid",
        "ANTHROPIC_API_KEY": "cpu-secret-api", "ANTHROPIC_AUTH_TOKEN": "cpu-secret-token",
        "OPENAI_API_KEY": "cpu-secret-openai", "OPENAI_BASE_URL": "https://cpu-openai.invalid",
        "HTTP_PROXY": "", "HTTPS_PROXY": "", "NO_PROXY": "*", "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8", "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "GIT_AUTHOR_NAME": "CPU", "GIT_AUTHOR_EMAIL": "cpu@example.invalid",
        "GIT_COMMITTER_NAME": "CPU", "GIT_COMMITTER_EMAIL": "cpu@example.invalid"}
    for key in tuple(os.environ):
        monkeypatch.delenv(key)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    facts = json.loads((root / "tests/fixtures/agate_b41_environment.json").read_text())
    snapshot = _snapshot("legacy", "cpu-fixture", EnvironmentQuery("L20D"), facts["detail"],
        facts["capabilities"], [], LegacyExecutor.capabilities.operations)
    monkeypatch.setattr(LegacyExecutor, "_environment_backend", lambda self: SimpleNamespace(
        backend="legacy", endpoint="https://cpu.invalid", operations=self.capabilities.operations,
        environment=self.request.environment if self.request else environment, ssh=False,
        check=lambda q, **kw: match_environment(q, snapshot), query=lambda q: snapshot))
    launches, effective = [], []
    def spawn(argv, **kwargs):
        assert kwargs["environment"]["PYTHONDONTWRITEBYTECODE"] == "1"
        launches.append(tuple(argv))
        return spawn_owned_session([sys.executable, "-B", str(child), *argv[3:]], **kwargs)
    monkeypatch.setattr("aka.execution.processes.spawn_owned_session", spawn)
    original = LegacyExecutor._run
    def run(self, command):
        if self.request is not None and command.submission is not None:
            effective.append(command)
            Path(environment["CPU_EFFECTIVE_COMMAND"]).write_text(json.dumps({
                "environment": dict(command.environment), "requirements": list(command.requirements)}))
        return original(self, command)
    monkeypatch.setattr(LegacyExecutor, "_run", run)
    return SimpleNamespace(root=root, task=task, requirements=requirements, service=service,
        launches=launches, effective=effective, environment=environment)


@pytest.mark.parametrize("entry", ["host", "cli"])
@pytest.mark.parametrize("enabled", [True, False], ids=["live-on", "live-off"])
def test_public_fresh_gemm_preparation_binds_before_intent(tmp_path, monkeypatch, entry, enabled):
    from aka.bootstrap.continuation import capture, restore
    from aka.bootstrap.host import run_selection
    from aka.bootstrap.profile import resolve_profile
    from aka.bootstrap.workflow_cli import main, profile_path
    from aka.contracts.startup import Invocation
    from aka.contracts.workflow import WorkflowSettings, CampaignRuntimeSettings
    from aka.contracts.workflow_launch import encode_launch
    from aka.workflow.legacy.requests import create_launch
    from aka.workflow.schema.launch import prepare_selection
    from aka.experiment.file_records import FileExperimentRecords
    h = native_cpu(tmp_path, monkeypatch)
    patches = [{"id": "resources", "config": {"plugins": [], "bundles": []}},
        {"id": "workflow", "config": {"startup": True, "search": {}, "inherit_state": False}},
        {"id": "evaluation", "config": {"shape_batch_size": 128, "comparison_method": "native-abba", "measurement_repetitions": 1}},
        {"id": "problem", "config": {"worker_requirements": str(h.requirements)}},
        {"id": "execution", "config": {"profile": "legacy", "patches": [{"id": "executor", "config": {
            "url": "https://cpu.invalid", "idempotent_replay": enabled}}]}}]
    config = {"format": "aka.schema.launch.v1", "limits": {"max_dispatches": 1}, "wiki_ids": []}
    if entry == "host":
        selected = resolve_profile(profile_path("light"), patches=patches, variables={"repo_root": str(tmp_path)})
        launch = create_launch(h.task, workspace_base=tmp_path / "runs", workspace_suffix="binding",
            settings=WorkflowSettings(platform="cuda", framework="triton", arch="sm_103", max_iters=1,
                target_util=0, setup_timeout=3600, framework_baseline="never", convert_after=0, fast_episodes=0, handoff_resumes=0),
            runtime=CampaignRuntimeSettings(hardware="L20D", timeout=3600, shape_batch_size=128))
        selected, invocation = prepare_selection(selected, Invocation(encode_launch(launch)), config)
        _, payload = capture(selected)
        assert run_selection(restore(payload), invocation) == 0
    else:
        patch = tmp_path / "patch.json"
        patch.write_text(json.dumps({"api_version": 1, "patches": patches}))
        schema = tmp_path / "schema.json"
        schema.write_text(json.dumps(config))
        assert main(["--repo-root", str(tmp_path), "--profile", "light", "--patch", str(patch),
            "--problem", str(h.task), "--workspace-base", str(tmp_path / "runs"), "--workspace-suffix", "binding",
            "--platform", "cuda", "--framework", "triton", "--arch", "sm_103", "--max-iters", "1",
            "--setup-timeout", "3600", "--framework-baseline", "never", "--convert-after", "0", "--fast-episodes", "0",
            "--handoff-resumes", "0", "--target-util", "0", "--hardware", "L20D", "--timeout", "3600",
            "--shape-batch-size", "128", "--schema", "--schema-config", str(schema)]) == 0
    workspace, = (tmp_path / "runs").glob("kernel_opt_*")
    command, = h.effective
    assert len(h.launches) == 1
    state, = (tmp_path / "runs").rglob("run-state.json")
    state = json.loads(state.read_bytes())
    assert state["counts"]["dispatches"] == 1 and state["counts"]["agent_calls"] == 0
    assert state["prepared"] is not None
    assert all(call["state"] == "settled" for call in state["calls"].values())
    records = FileExperimentRecords(workspace.parent / ".aka-experiments", str(workspace.absolute()), create=False)
    record, = records.iter_records()
    intent, = (json.loads(records.read_artifact(a)) for a in record.artifacts if a.role == "submission_intent")
    assert intent["environment_sha256"] == digest(dict(command.environment))
    assert intent["submission_sha256"] == hashlib.sha256(command.submission).hexdigest()
    assert intent["idempotency_key"] == command.idempotency_key
    assert intent["timeout"] == command.timeout == 3600
    assert intent["operation"] == command.operation == "run"
    assert record.attempts[0].disposition == "terminal"
    service = json.loads(h.service.read_bytes())
    assert service["command_environment_sha256"] == intent["environment_sha256"]
    launcher = dict(command.environment)
    if command.requirements:
        launcher["AKA_GATEWAY_DEV_DEPENDENCIES"] = json.dumps(command.requirements)
    assert service["launcher_environment_sha256"] == digest(launcher)
    posts = [c for c in service["calls"] if c["method"] == "POST"]
    assert len(posts) == 1 and posts[0]["payload"]["idempotency_key"] == command.idempotency_key
    assert posts[0]["payload"].get("env_vars", {}) == {}
    wire = json.dumps(posts[0]["payload"])
    assert all(value not in wire for value in ("cpu-secret-api", "cpu-secret-token", "cpu-secret-openai"))
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "PATH", "PYTHONDONTWRITEBYTECODE"):
        assert command.environment[name] == h.environment[name]


@pytest.mark.parametrize("scoped", [False, True], ids=["host-context", "restricted-context"])
def test_binding_precedence_is_idempotent_and_context_is_immutable(tmp_path, scoped):
    from aka.contracts.execution import BackendCapabilities, CommandSpec, ExecutionContext, JobResult
    from aka.contracts.filesystem import FilesystemAccess, FilesystemMount
    from aka.execution.binding import SandboxExecutor
    scope = FilesystemAccess((FilesystemMount(tmp_path, tmp_path, True),)) if scoped else None
    host = {"HOST_ONLY": "host", "CALLER": "host", "ATREX_SANDBOX_REQUIREMENTS_FILE": "host-requirements",
            "ATREX_PRIVATE_REFERENCE_DIR": "host-reference"}
    context = ExecutionContext(host, "L20D", filesystem_access=scope)
    captured = []
    def run(command):
        captured.append(command)
        return JobResult(0)
    executor = SandboxExecutor(SimpleNamespace(
        capabilities=BackendCapabilities(filesystem_isolation=True, cancellation=True), run=run), context)
    host["HOST_ONLY"] = "changed-after-context"
    command = CommandSpec(tmp_path, ("true",), environment={"CALLER": "caller",
        "ATREX_SANDBOX_REQUIREMENTS_FILE": "caller-requirements", "ATREX_PRIVATE_REFERENCE_DIR": "caller-reference"})
    bound = executor.bind_command(command)
    assert executor.bind_command(bound) is bound
    assert bound.environment["CALLER"] == "caller" and bound.hardware == "L20D"
    assert bound.environment["ATREX_SANDBOX_REQUIREMENTS_FILE"] == ("caller-requirements" if scoped else "host-requirements")
    assert bound.environment["ATREX_PRIVATE_REFERENCE_DIR"] == ("caller-reference" if scoped else "host-reference")
    assert bound.environment.get("HOST_ONLY") == (None if scoped else "host")
    with pytest.raises(AttributeError):
        executor.context = replace(context, environment={"HOST_ONLY": "drift"})
    with pytest.raises(TypeError):
        context.environment["HOST_ONLY"] = "drift"
    # Observation and cancellation wrappers copy CommandSpec; binding still
    # preserves the effective identity and hands the same object to execution.
    from threading import Event
    observed = replace(bound, on_observation=lambda result: None, cancel_event=Event())
    assert executor.bind_command(observed) is observed
    executor.run(observed)
    assert captured == [observed] and captured[0] is observed
