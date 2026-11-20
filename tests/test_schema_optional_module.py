"""A3 CPU/synthetic tests over installed metadata, actual tools and native Agent.

No test invokes a model, CUDA, Agate, native audits or claims qualification.
"""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from uuid import UUID

import pytest

from aka.agent.provider import Agent
from aka.agent.tools.client import call_tool, ToolSessionError
from aka.task.candidate_workspace.provider import GitCandidateWorkspace
from aka.workflow.schema.context import RoleContext
from aka.workflow.schema.provider import SchemaSelection
from aka.workflow.schema.release import SchemaUnavailable, verify_release
from aka.workflow.schema.session import native_agent, RoleSession
from kernelschema.api import prepare_run
from kernelschema.artifacts import Bundle, Publication
from kernelschema.contracts import ProtocolError, UnsupportedError, jcs, sha256
from kernelschema.evaluation import freeze_task
from kernelschema.extract_bridge import consume_raw
from kernelschema.resources import checkout_root
from kernelschema.runtime import ClaudeProfile


@pytest.fixture
def fixed(tmp_path):
    path = checkout_root() / "tests/workflows/fixtures/v1/row_softmax/build.py"
    spec = importlib.util.spec_from_file_location("a3_shared_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    f = fixture.build_bundle(tmp_path / "bundle")
    p = Publication(f["root"])
    task = freeze_task(p, environment=f["originals"]["task/environment.md"])
    schema = consume_raw(p, checkout_root() / "examples/workflows/row_softmax/schema", task=task["task"])["schema"]
    native_profile = profile(tmp_path)
    selected_profile = {**Bundle(f["root"]).document(f["profile"], "backend_profile"),
                        "parameters": {"claude": native_profile.as_dict()}, "model": native_profile.requested_model}
    profile_ref = p.put_json("a3-native-profile.json", selected_profile)
    request = {**f["request"], "task": task["task"], "schema": schema,
               "backend": {"name": "synthetic", "profile": profile_ref}}
    request_ref = p.put_json("request.json", request)
    p.commit([request_ref])
    (f["root"] / "request.json").write_bytes(jcs(request))
    run = prepare_run("implement", request=f["root"] / "request.json", out=tmp_path / "run",
                      backend_identity={"synthetic": "A3 CPU native boundary only"})
    return SimpleNamespace(run=run, fixture=f, task=task)


def profile(tmp_path):
    binary = tmp_path / "fake-native"
    binary.write_text("synthetic native executable bytes; never a model\n")
    return ClaudeProfile(str(binary), "synthetic", sha256(binary.read_bytes()), "synthetic-model",
                         ("synthetic-model",), timeout_ms=5000)


def fake_runner(trace, *, transform=None):
    def run(command, *, cwd, timeout, env):
        trace.append(("exec", command, dict(env)))
        assert command[1:3] == ["-m", "aka.workflow.schema.stdin_exec"]
        data = Path(command[3]).read_bytes()
        assert sha256(data) == command[4]
        payload = json.loads(data)
        argv = command[command.index("--") + 1:]
        session = argv[argv.index("--session-id") + 1]
        init = {"type": "system", "subtype": "init", "tools": [], "mcp_servers": [], "skills": [],
                "plugins": [], "permissionMode": "dontAsk", "cwd": str(cwd),
                "session_id": session, "model": "synthetic-model"}
        output = {"protocol": "ks.claude.output.v1", "role": payload["role"],
                  "assignment_sha256": payload["assignment_sha256"], "status": "ok",
                  "kind": "candidate_source", "content": "# α\r\ndef kernel():\r\n    return 7", "reason": "synthetic"}
        terminal = {"type": "result", "subtype": "success", "is_error": False,
                    "session_id": session, "result": json.dumps(output),
                    "usage": {"input_tokens": 5, "output_tokens": 4, "cache_read_input_tokens": 0,
                              "cache_creation_input_tokens": 0}}
        events = [init, terminal]
        if transform:
            transform(events)
        return "\n".join(json.dumps(item) for item in events), "", 0, False
    return run


def test_fixed_role_native_first_repair_retry_and_reopen(fixed, tmp_path):
    p = profile(tmp_path)
    role = RoleContext.prepare(fixed.run, "coder", source=fixed.task["starter"])
    retained = tmp_path / "retained"
    role.retain(retained)
    reopened = RoleContext.reopen(fixed.run, retained, expected_sha256=sha256(role.payload))
    assert reopened.payload == role.payload
    assert {item["id"] for item in role.document["materials"]}.isdisjoint({"material/workflow_objective"})
    assert "Implement only the fixed input Schema" in role.document["fixed_assignment"]["instruction"]
    trace = []
    agent = native_agent(p, process_runner=fake_runner(trace), environment={"HOME": str(Path.home()), "PATH": os.environ["PATH"]})
    try:
        for index, purpose in enumerate(("initial", "repair", "rate_limit_retry", "resume", "reopen")):
            scope = tmp_path / f"scope-{index}"
            scope.mkdir()
            def authorized():
                trace.append(("authorize",))
                return True
            session = RoleSession(agent, reopened, p, workspace=scope, session_id=str(UUID(int=index + 1)),
                output_directory=tmp_path / f"observations-{index}", authorize=authorized,
                record_locator=lambda value: trace.append(("locator", value)))
            try:
                result = session.invoke("Explore the wiki freely and rewrite Schema", purpose=purpose)
                assert result.permission_errors == ()
                assert result.output.candidate_source == b"# \xce\xb1\r\ndef kernel():\r\n    return 7"
                assert result.invocation.terminal_usage.total_tokens == 9
                draft = tmp_path / f"draft-{index}"
                draft.mkdir()
                candidate = GitCandidateWorkspace()
                candidate.install_source(draft, result.native_source())
                assert candidate.read_source(draft).files["kernel.py"] == result.output.candidate_source
                binding = result.freeze_source(Publication(fixed.run.root), GitCandidateWorkspace())
                assert fixed.run.bundle.source(binding.source) == result.output.candidate_source
                assert binding.kernel_sha256 == sha256(result.output.candidate_source)
                changed = RoleContext.prepare(fixed.run, "coder", source=binding.source)
                assert changed.assignment_sha256 == role.assignment_sha256
                assert changed.document["source_sha256"] == binding.source["sha256"]
                assert changed.payload != role.payload
                with pytest.raises(UnsupportedError):
                    session.invoke(resume=True)
            finally:
                session.close()
        assert [item[0] for item in trace] == ["locator", "authorize", "exec"] * 5
        calls = [item for item in trace if item[0] == "exec"]
        for _, command, environment in calls:
            assert Path(command[3]).read_bytes() == role.payload
            assert "dangerously-skip-permissions" not in " ".join(command)
            assert command[command.index("--tools") + 1] == ""
            assert command[command.index("--mcp-config") + 1] == '{"mcpServers":{}}'
            assert "--no-session-persistence" in command and "--safe-mode" in command
            assert "ATREX_SESSION_SETTINGS" not in environment
    finally:
        agent.close()


@pytest.mark.parametrize("bad", ["tool", "schema_output", "missing_init", "model", "terminal_model"])
def test_untrusted_native_permission_claims_preserve_usage_and_reject_output(fixed, tmp_path, bad):
    def corrupt(events):
        if bad == "tool":
            events[0]["tools"] = ["Bash", "Read", "Edit"]
        elif bad == "schema_output":
            output = json.loads(events[-1]["result"])
            output.update(kind="writing_draft", content={"path": "contract.yaml", "text": "weaken acceptance"})
            events[-1]["result"] = json.dumps(output)
        elif bad == "missing_init":
            events.pop(0)
        elif bad == "model":
            events[0]["model"] = "another-model"
        else:
            events[-1]["modelUsage"] = {"another-model": {"outputTokens": 4}}
    p, trace = profile(tmp_path), []
    agent = native_agent(p, process_runner=fake_runner(trace, transform=corrupt))
    scope = tmp_path / "scope"
    scope.mkdir()
    session = RoleSession(agent, RoleContext.prepare(fixed.run, "coder", source=fixed.task["starter"]), p, workspace=scope,
        session_id=str(UUID(int=1)), output_directory=tmp_path / "output", authorize=lambda: True,
        record_locator=lambda _: None)
    try:
        result = session.invoke()
        assert result.output is None and result.permission_errors
        assert result.invocation.terminal_usage.total_tokens == 9
        with pytest.raises(ProtocolError):
            result.freeze_source(Publication(fixed.run.root), GitCandidateWorkspace())
    finally:
        session.close()
        agent.close()


def test_owner_denial_and_unrestricted_agent_cannot_launch(fixed, tmp_path):
    p, trace = profile(tmp_path), []
    scope = tmp_path / "scope"
    scope.mkdir()
    role = RoleContext.prepare(fixed.run, "coder", source=fixed.task["starter"])
    original = Agent()
    try:
        with pytest.raises(PermissionError, match="restricted"):
            RoleSession(original, role, p, workspace=scope, session_id=str(UUID(int=1)), output_directory=tmp_path / "out",
                        authorize=lambda: True, record_locator=lambda _: None)
    finally:
        original.close()
    agent = native_agent(p, process_runner=fake_runner(trace))
    session = RoleSession(agent, role, p, workspace=scope, session_id=str(UUID(int=1)), output_directory=tmp_path / "out",
                          authorize=lambda: False, record_locator=lambda value: trace.append(("locator", value)))
    try:
        raw_session = agent.open_session("claude", scope, session_id=str(UUID(int=2)), output_directory=tmp_path / "bypass")
        try:
            with pytest.raises(PermissionError, match="fixed native role input"):
                raw_session.invoke("bypass the Schema role context", reasoning_effort=p.effort)
        finally:
            raw_session.close()
        with pytest.raises(PermissionError, match="owner denied"):
            session.invoke()
        assert [item[0] for item in trace] == ["locator"]
    finally:
        session.close()
        agent.close()


def test_fixed_request_rejects_another_native_profile_before_launch(fixed, tmp_path):
    wrong = replace(profile(tmp_path), requested_model="other-model", reported_models=("other-model",))
    trace = []
    agent = native_agent(wrong, process_runner=fake_runner(trace))
    scope = tmp_path / "scope"
    scope.mkdir()
    try:
        with pytest.raises(ProtocolError, match="profile/model differs"):
            RoleSession(agent, RoleContext.prepare(fixed.run, "coder", source=fixed.task["starter"]), wrong,
                workspace=scope, session_id=str(UUID(int=1)), output_directory=tmp_path / "out",
                authorize=lambda: True, record_locator=lambda _: None)
        assert trace == []
    finally:
        agent.close()


def test_existing_ks_owner_authorizes_the_same_native_operation(fixed, tmp_path):
    from aka.workflow.schema.authority import NativeCallAuthority
    from kernelschema.budget import OwnerAccount, cost
    p, trace = profile(tmp_path), []
    role = RoleContext.prepare(fixed.run, "coder", source=fixed.task["starter"])
    owner = OwnerAccount.for_context(fixed.run, create=True)
    reservation = owner.reserve(fixed.run.run_id, "native-coder", role="coder", purpose="A3 synthetic native call",
                                amount=cost(role="coder", calls=1, active_ms=8000))
    authority = NativeCallAuthority.bind(fixed.run, owner, reservation, source=fixed.task["starter"])
    with pytest.raises(ProtocolError, match="persisted"):
        authority.authorize()
    with pytest.raises(ProtocolError, match="another original"):
        NativeCallAuthority.bind(fixed.run, owner, {**reservation, "run_id": "wrong"}, source=fixed.task["starter"])
    scope = tmp_path / "scope"
    scope.mkdir()
    agent = native_agent(p, process_runner=fake_runner(trace))
    session = RoleSession.from_authority(agent, role, p, authority=authority, workspace=scope,
        session_id=str(UUID(int=1)), output_directory=tmp_path / "native-output")
    try:
        result = session.invoke()
        assert result.output is not None
        assert owner.operations(fixed.run.run_id)["native-coder"]["launched"]
        intent = fixed.run.intents()[0]
        assert intent["reservation"] == reservation
        assert intent["native_locator"]["source_sha256"] == fixed.task["starter"]["sha256"]
        assert float(trace[0][2]["ATREX_AGENT_TIMEOUT_SECONDS"]) <= 5
        # The test's synthetic terminal is explicitly absorbed by the original
        # KS owner. A3 provides no independent usage accounting implementation.
        row = {"producer": "a3-synthetic", "operation_id": "native-coder", "role": "coder",
            "purpose": "A3 synthetic native call", "receipt_id": "synthetic-native-result", "sequence": 0,
            "receipt": None, "state": "settled", "calls": 1, "evaluator_jobs": 0,
            "tokens": result.invocation.terminal_usage.total_tokens, "gpu_ms": 0}
        assert owner.absorb(fixed.run.run_id, row, terminated=True)
        assert not owner.absorb(fixed.run.run_id, row, terminated=True)
        assert set(owner.operations(fixed.run.run_id)) == {"native-coder"}
    finally:
        session.close()
        agent.close()


def test_actual_toolhost_enforces_role_ids_and_no_path_or_write(fixed, tmp_path):
    role = RoleContext.prepare(fixed.run, "coder")
    reviewer = RoleContext.prepare(fixed.run, "reviewer")
    scope = tmp_path / "scope"
    scope.mkdir()
    agent = Agent()
    try:
        with role.tools(agent, scope) as grant:
            def call(value):
                return json.loads(call_tool("schema", jcs(value), workspace=scope, environment=grant.environment))
            assert call({"op": "read", "id": "task/contract"})["category"] == "assignment"
            assert "CONTRACT" in call({"op": "effective", "node": "root"})
            assert call({"op": "check"})["structure"] == "pass"
            for request in ({"op": "read", "id": "material/workflow_objective"},
                            {"op": "read", "id": "evaluator/golden.py"},
                            {"op": "read", "id": str(fixed.run.root / "request.json")},
                            {"op": "write", "id": "task/contract", "text": "weaken"},
                            {"op": "write", "id": "schema/root", "text": "replace"},
                            {"op": "effective", "node": "root", "path": "/tmp/evil"}):
                with pytest.raises(ToolSessionError):
                    call(request)
            old_environment = dict(grant.environment)
        with pytest.raises(ToolSessionError):
            call_tool("schema", jcs({"op": "check"}), workspace=scope, environment=old_environment)
        with reviewer.tools(agent, scope) as grant:
            result = json.loads(call_tool("schema", jcs({"op": "read", "id": "material/workflow_objective"}),
                                          workspace=scope, environment=grant.environment))
            assert result["category"] == "role_material"
        corrupted = role.document
        corrupted["materials"][0]["text"] = "replacement Task criteria"
        with pytest.raises(ProtocolError, match="actual Materials"):
            replace(role, payload=jcs(corrupted)).validate()
        from kernelschema.runtime import RoleDelivery
        private = fixed.run.objects["evaluator"]["entrypoint"]
        with pytest.raises(ProtocolError, match="private evaluator"):
            RoleContext.prepare(fixed.run, "coder", deliveries=(RoleDelivery("renamed-reference", private, ("coder",)),))
    finally:
        agent.close()


def test_nonroot_role_reader_retains_actual_root_contract(fixed, tmp_path):
    from copy import deepcopy
    import yaml
    p = Publication(fixed.fixture["root"])
    schema = deepcopy(fixed.run.objects["schema"])
    root = next(item["file"] for item in schema["raw"] if item["node"] == "root")
    original = yaml.safe_load(fixed.run.bundle.read(root))
    child = deepcopy(original)
    child.pop("CONTRACT")
    child["META_DATA"]["Dispatch"] = {"id": "branch", "when": "Same selected shapes"}
    child["DESIGN"] = {region: {"from": "root"} for region in ("Math_reshape", "Work_schedule", "Tile_pipeline")}
    raw = p.put_bytes("raw/branch.yaml", yaml.safe_dump(child, sort_keys=False).encode())
    dispatch = p.put_bytes("raw/dispatch.md", fixed.run.bundle.read(schema["dispatch"]).replace(
        b"| [root]", b"| [branch](branch.yaml) | same | root |\n| [root]", 1))
    schema.update(raw=[{"node": "branch", "file": raw}, *schema["raw"]], dispatch=dispatch)
    schema_ref = p.put_json("schema.json", schema)
    request = {**fixed.run.request, "schema": schema_ref}
    reference = p.put_json("request.json", request)
    p.commit([reference])
    path = fixed.fixture["root"] / "branch-request.json"
    path.write_bytes(jcs(request))
    run = prepare_run("implement", request=path, out=tmp_path / "branch-run", backend_identity={"synthetic": "A3 branch reader"})
    scope = tmp_path / "branch-scope"
    scope.mkdir()
    agent = Agent()
    try:
        with RoleContext.prepare(run, "coder").tools(agent, scope) as grant:
            result = json.loads(call_tool("schema", jcs({"op": "effective", "node": "branch"}),
                                         workspace=scope, environment=grant.environment))
            assert result["CONTRACT"] == original["CONTRACT"]
            assert result["DESIGN"] == original["DESIGN"]
    finally:
        agent.close()


def test_selection_uses_a2_problem_and_checks_backend_before_import(fixed, tmp_path, monkeypatch):
    config = tmp_path / "selection.json"
    target = tmp_path / "synthetic_backend.py"
    marker = tmp_path / "factory-executed"
    target.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\ndef create_workflow(selection, modules): return None\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    value = {"format": "aka.schema.selection.v1", "request": str(fixed.fixture["root"] / "request.json"),
             "run_root": "next-run", "backend": {"factory": "synthetic_backend:create_workflow",
                "implementation_sha256": "sha256:" + "0" * 64, "parameters": {}}}
    config.write_bytes(jcs(value))
    selection = SchemaSelection.load(config)
    provider = selection.problem_provider()
    problem = provider.load(fixed.fixture["root"], snapshot_directory=tmp_path / "snapshot")
    assert {item.path for item in problem.agent.files} == {"contract.yaml", "acceptance.md"}
    assert problem.initial_source.kind == "kernel" and provider.runtime_files(problem) == ()
    with pytest.raises(SchemaUnavailable, match="differs"):
        selection.backend_factory()
    assert not marker.exists()
    value["backend"]["implementation_sha256"] = sha256(target.read_bytes())
    config.write_bytes(jcs(value))
    assert callable(SchemaSelection.load(config).backend_factory())
    assert marker.exists()


def test_on_composition_selects_real_restricted_agent_and_problem(fixed, tmp_path, monkeypatch):
    from aka.bootstrap import host
    from aka.bootstrap.workflow_cli import profile_path
    from aka.contracts.startup import Invocation
    from aka.workflow.schema.launch import run
    from aka.workflow.schema import agent_plugin, problem_plugin
    from aka.workflow.schema.session import SchemaNativeAgent
    p = profile(tmp_path)
    publication = Publication(fixed.fixture["root"])
    backend_profile = {**fixed.run.objects["profile"], "parameters": {"claude": p.as_dict()}, "model": p.requested_model}
    ref = publication.put_json("native-profile.json", backend_profile)
    request = {**fixed.run.request, "backend": {"name": "synthetic", "profile": ref}}
    request_ref = publication.put_json("native-request.json", request)
    publication.commit([request_ref])
    request_path = fixed.fixture["root"] / "native-request.json"
    request_path.write_bytes(jcs(request))
    target = tmp_path / "a3_factory.py"
    target.write_text("def create_workflow(selection, modules): return None\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    config = tmp_path / "selection.json"
    config.write_bytes(jcs({"format": "aka.schema.selection.v1", "request": str(request_path),
        "run_root": "new-native-run", "backend": {"factory": "a3_factory:create_workflow",
            "implementation_sha256": sha256(target.read_bytes()), "parameters": {}}}))
    captured = []
    monkeypatch.setattr(host, "run_selection", lambda selection, invocation: captured.append(selection) or 0)
    assert run(profile_path("light"), Invocation(()), configuration=config, variables={"repo_root": str(tmp_path)}) == 0
    composition = captured[0].composition
    assert composition.row("workflow").name == "aka.workflow.schema.plugin"
    assert composition.row("resources").config == {"plugins": [], "bundles": []}
    patch = tmp_path / "sandbox-patch.json"
    patch.write_text(json.dumps({"api_version": 1, "patches": [
        {"id": "agent", "config": {"sandbox": "bwrap", "proc_mode": "private"}}]}))
    with pytest.raises(ValueError, match="plain Light Agent"):
        run(profile_path("light"), Invocation(()), configuration=config, patch_files=[patch],
            variables={"repo_root": str(tmp_path)})
    services, cleanup = {}, []
    ctx = SimpleNamespace(effect=lambda fn, **kwargs: cleanup.append(fn),
                          provide=lambda key, value: services.setdefault(key.name, value))
    try:
        agent_plugin.apply(ctx, composition.row("agent").config)
        problem_plugin.apply(ctx, composition.row("problem").config)
        assert isinstance(services["agent"], SchemaNativeAgent)
        assert services["problem"].task == fixed.task["task"]
        assert services["agent"].schema_adapter.profile.as_dict() == p.as_dict()
        config.write_bytes(config.read_bytes() + b"\n")
        with pytest.raises(SchemaUnavailable, match="selection changed"):
            agent_plugin.apply(ctx, composition.row("agent").config)
    finally:
        for close in cleanup:
            close()


def test_fresh_child_installed_metadata_and_resource_source_pair(tmp_path):
    code = """from importlib import metadata
from aka.workflow.schema.release import verify_release
import json, sys
value = verify_release()
value['schema_requires'] = metadata.requires('atrex-aka-schema')
value['python'] = sys.version_info[:2]
print(json.dumps(value))
"""
    child = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path,
                           capture_output=True, text=True, check=True)
    value = json.loads(child.stdout)
    assert value["ks_commit"] == verify_release()["ks_commit"]
    assert Path(value["ks_module"]).parent.parent == Path(value["ks_root"])
    assert "atrex-aka-light==0.2.0" in value["schema_requires"]


def test_wrong_ks_package_is_rejected_before_its_initializer_runs(tmp_path):
    fake = tmp_path / "kernelschema"
    fake.mkdir()
    marker = tmp_path / "initializer-ran"
    (fake / "__init__.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n")
    code = """import sys
sys.path.insert(0, sys.argv[1])
from aka.workflow.schema.release import verify_release, SchemaUnavailable
try:
    verify_release()
except SchemaUnavailable:
    print('unverified KS rejected before import')
else:
    raise AssertionError('wrong KS accepted')
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, str(tmp_path)], cwd=tmp_path,
                             capture_output=True, text=True, check=True)
    assert "rejected before import" in result.stdout and not marker.exists()


def test_stdin_exec_preserves_exact_payload_in_real_child(tmp_path):
    raw = b'{"text":"\\r\\n exact UTF-8 \xce\xb1"}'
    path = tmp_path / "input.json"
    path.write_bytes(raw)
    command = [sys.executable, "-I", "-m", "aka.workflow.schema.stdin_exec", str(path), sha256(raw),
               sha256(Path(sys.executable).read_bytes()), "--", sys.executable, "-I", "-c",
               "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"]
    child = subprocess.run(command, cwd=tmp_path, capture_output=True, check=True)
    assert child.stdout == raw
    command[6] = "sha256:" + "0" * 64
    rejected = subprocess.run(command, cwd=tmp_path, capture_output=True)
    assert rejected.returncode != 0 and rejected.stdout == b""


@pytest.mark.parametrize("switch", [[], ["--no-schema"], ["--no-schema", "--schema-config", "/nonexistent/KernelSchema/config.json"]])
def test_original_cli_off_with_optional_import_io_and_network_forbidden(tmp_path, switch):
    code = r'''
import importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.startswith(('kernelschema', 'aka.workflow.schema')):
            raise AssertionError('optional import while off: ' + fullname)
sys.meta_path.insert(0, Block())
def audit(event, args):
    if event in {'open', 'os.listdir', 'os.scandir'} and args:
        if 'KernelSchema' in str(args[0]) or '/optional/' in str(args[0]):
            raise AssertionError('optional filesystem while off')
    if event == 'socket.connect': raise AssertionError('network while off')
sys.addaudithook(audit)
from aka.bootstrap import workflow_cli
def original(profile, invocation, **options):
    print(json.dumps({'profile': str(profile), 'argv': invocation.argv, 'options': options}, default=str))
    return 0
workflow_cli.run_profile = original
raise SystemExit(workflow_cli.main(sys.argv[1:]))
'''
    result = subprocess.run([sys.executable, "-I", "-c", code, "--repo-root", str(tmp_path),
         "--problem", str(tmp_path), "--platform", "cpu", "--framework", "triton", *switch],
         cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "light.json" in json.loads(result.stdout)["profile"]
    if switch:
        default = subprocess.run([sys.executable, "-I", "-c", code, "--repo-root", str(tmp_path),
             "--problem", str(tmp_path), "--platform", "cpu", "--framework", "triton"],
             cwd=tmp_path, capture_output=True, text=True, check=True)
        assert result.stdout == default.stdout


def test_original_cli_on_missing_configuration_is_clear_and_zero_work(tmp_path):
    child = subprocess.run([sys.executable, "-I", "-m", "aka", "workflow", "--repo-root", str(tmp_path),
        "--problem", str(tmp_path), "--platform", "cpu", "--framework", "triton", "--schema"],
        cwd=tmp_path, capture_output=True, text=True)
    assert child.returncode == 2
    assert "--schema requires --schema-config" in child.stderr
