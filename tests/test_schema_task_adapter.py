"""Real AKA Task/Source/Evaluator/Experiment objects, KS fixed full CPU evaluator.

CPU timing/postconditions are logic evidence, never CUDA or native ABBA evidence.
"""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from aka.contracts.evaluation import EvaluationConfiguration, EvaluationRequest
from aka.contracts.execution import UnsupportedExecution, validate_executor
from aka.contracts.workspace import SourceRequest
from aka.experiment.runner import LegacyExperiment
from aka.task.candidate_workspace.content import install_files
from aka.task.candidate_workspace.provider import GitCandidateWorkspace
from aka.task.source.legacy import LegacySourceProvider
from aka.workflow.schema.evaluation import (EvaluationOperation, LocalChannel, SchemaExecution,
    SchemaEvaluatorFactory, verify_report_binding, require_applicable, native_verification)
from aka.workflow.schema.source import bind_source, freeze_native_source, selected_seed
from aka.workflow.schema.task import SchemaProblemProvider, evaluation_bundle
from kernelschema.artifacts import Bundle, Publication
from kernelschema.contracts import PROTOCOL, ProtocolError, UnsupportedError, jcs, sha256
from kernelschema.evaluation import freeze_task, LocalExecutor
from kernelschema.extract_bridge import consume_raw
from kernelschema.resources import checkout_root


@pytest.fixture(scope="module")
def task_bundle(tmp_path_factory):
    # Execute the shared pinned fixture generator; do not transcribe Task criteria.
    path = checkout_root() / "tests/workflows/fixtures/v1/row_softmax/build.py"
    spec = importlib.util.spec_from_file_location("shared_row_softmax_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    root = tmp_path_factory.mktemp("shared-task")
    f = fixture.build_bundle(root)
    p = Publication(root)
    complete = freeze_task(p, environment=f["originals"]["task/environment.md"])
    schema = consume_raw(p, checkout_root() / "examples/workflows/row_softmax/schema", task=complete["task"])["schema"]
    p.commit([complete["task"], schema], subjects=[complete["task"]])
    return dict(f, **complete, schema=schema, bundle=Bundle(root), scope=f["request"]["scope"])


@pytest.fixture
def native_task(task_bundle, tmp_path):
    f = task_bundle
    provider = SchemaProblemProvider(f["task"], f["scope"], f["request"]["materials"])
    problem = provider.load(f["root"], snapshot_directory=tmp_path / "private/task")
    workspace = tmp_path / "agent"
    workspace.mkdir()
    install_files(workspace, problem.agent.files)
    # Existing SourceProvider and actual InitialSource, without Git mutation/config.
    request = SourceRequest(kind="kernel", initial=problem.initial_source)
    content = LegacySourceProvider().prepare(request)
    install_files(workspace, content.files)
    return f, provider, problem, workspace


def configured(native_task, tmp_path, *, name="one", expected=None):
    f, provider, problem, workspace = native_task
    execution = SchemaExecution(tmp_path / "host-operations", LocalChannel(LocalExecutor(tmp_path / "jobs", python=sys.executable)))
    operation = EvaluationOperation(name, "stable-" + name, "owner-a2",
        {"owner": "owner-a2", "evaluator_jobs": 1, "active_ms": 60000}, expected)
    factory = SchemaEvaluatorFactory(execution=execution, operation=operation)
    evaluator = factory.create(EvaluationConfiguration(timeout=60), candidate_workspace=GitCandidateWorkspace(),
                               evaluation_input=problem.evaluation)
    return execution, evaluator, EvaluationRequest(workspace, mode="schema_task")


def test_public_initial_private_snapshot_roles(native_task, tmp_path):
    f, provider, problem, workspace = native_task
    assert {x.path for x in problem.agent.files} == {"contract.yaml", "acceptance.md"}
    assert provider.runtime_files(problem) == ()
    provider.content(problem, workspace).assert_private()
    assert (workspace / "kernel.py").read_bytes() == f["bundle"].source(f["baseline"])
    assert problem.initial_source.kind == "kernel"
    assert selected_seed(f["bundle"], f["task"], None) == f["starter"]
    assert f["bundle"].source(f["starter"]) != (workspace / "kernel.py").read_bytes()
    exported = provider.export_snapshot(problem, tmp_path / "restored")
    restored = provider.validate_snapshot(exported)
    assert evaluation_bundle(restored.evaluation)[1]["task"] == f["task"]
    with pytest.raises(ValueError, match="source configuration"):
        replace(provider, role="reviewer").load(f["root"], snapshot_directory=problem.snapshot_directory)
    (workspace / "inputs.json").write_text("private")
    with pytest.raises(ProtocolError, match="leaked"):
        provider.content(problem, workspace).assert_private()


def test_private_material_alias_and_subset_fail_before_snapshot(task_bundle, tmp_path):
    f = task_bundle
    p = Publication(f["root"])
    evaluator = f["bundle"].document(f["evaluator"], "evaluator")
    private = evaluator["entrypoint"]
    alias = p.put_bytes("renamed-allowed.txt", f["bundle"].read(private))
    materials = p.put_json("materials.json", {"protocol": PROTOCOL, "kind": "materials", "entries": [
        {"id": "mislabelled", "file": alias, "roles": ["coder"], "origin": "test", "license": "test"}]})
    partial = p.put_json("partial.json", {**f["bundle"].document(f["scope"], "scope"), "coverage_ids": ["shape_m1_n128"],
        "bindings": [{"node": "root", "variant": None, "coverage_ids": ["shape_m1_n128"]}]})
    p.commit([materials, partial])
    with pytest.raises(ProtocolError, match="private evaluator"):
        SchemaProblemProvider(f["task"], f["scope"], materials).load(f["root"], snapshot_directory=tmp_path / "private")
    with pytest.raises(UnsupportedError, match="full Task"):
        SchemaProblemProvider(f["task"], partial).load(f["root"], snapshot_directory=tmp_path / "partial")
    assert not (tmp_path / "private").exists() and not (tmp_path / "partial").exists()


def test_unimplemented_evaluator_and_wrong_baseline_fail_before_bootstrap(task_bundle, tmp_path):
    f = task_bundle
    with pytest.raises(UnsupportedError, match="implementation/version"):
        SchemaProblemProvider(f["request"]["task"], f["scope"]).load(
            f["root"], snapshot_directory=tmp_path / "interface-only")
    p = Publication(f["root"])
    wrong = p.put_json("wrong-baseline.json", {**f["bundle"].document(f["task"], "task"),
                                              "baseline_source": f["starter"]})
    p.commit([wrong])
    with pytest.raises(ProtocolError, match="baseline differs"):
        SchemaProblemProvider(wrong, f["scope"]).load(f["root"], snapshot_directory=tmp_path / "wrong")
    assert not (tmp_path / "interface-only").exists() and not (tmp_path / "wrong").exists()


@pytest.mark.parametrize("data", [b"# CRLF\r\npass\r\n", b"pass", "# 非 ASCII\npass\n".encode()])
def test_exact_source_bytes_native_identity(data, tmp_path):
    workspace = tmp_path / "native"
    workspace.mkdir()
    (workspace / "kernel.py").write_bytes(data)
    (workspace / "kernel.py").chmod(0o644)
    native = GitCandidateWorkspace()
    p = Publication(tmp_path / "bundle")
    binding = freeze_native_source(p, workspace, native)
    p.commit([binding.source])
    bundle = Bundle(p.root)
    assert bundle.source(binding.source) == data
    assert binding.native_identity == sha256(data)[7:]
    assert binding.kernel_sha256 == sha256(data) and binding.size == len(data)
    assert bind_source(bundle, binding.source, native) == binding
    (workspace / "solution.json").write_text(json.dumps({"sources": [{"path": "kernel.py"}], "note": "not part of C"}))
    p2 = Publication(tmp_path / "second")
    assert freeze_native_source(p2, workspace, native).source["sha256"] == binding.source["sha256"]


def test_bad_seed_missing_multifile_mode_and_encoding(task_bundle, tmp_path):
    f = task_bundle
    with pytest.raises(ProtocolError):
        selected_seed(f["bundle"], f["task"], {**f["starter"], "sha256": "sha256:" + "0" * 64})
    workspace = tmp_path / "native"
    workspace.mkdir()
    native = GitCandidateWorkspace()
    def freeze():
        return freeze_native_source(Publication(tmp_path / "p"), workspace, native)
    with pytest.raises((ProtocolError, ValueError)):
        freeze()
    for data in (b"\xef\xbb\xbfpass", b"\xff"):
        (workspace / "kernel.py").write_bytes(data)
        with pytest.raises(ProtocolError):
            freeze()
    (workspace / "kernel.py").write_bytes(b"pass")
    (workspace / "kernel.py").chmod(0o755)
    with pytest.raises(ProtocolError, match="0644"):
        freeze()
    (workspace / "kernel.py").chmod(0o644)
    (workspace / "solution.json").write_text('{"sources":["kernel.py","helper.py"]}')
    (workspace / "helper.py").write_bytes(b"pass")
    with pytest.raises(ProtocolError, match="auxiliary"):
        freeze()
    (workspace / "solution.json").unlink()
    (workspace / "kernel.py").write_bytes(b"import helper\n")
    with pytest.raises(ProtocolError, match="auxiliary"):
        freeze()


def test_public_source_extra_file_and_expected_C_rejected(task_bundle, tmp_path):
    f = task_bundle
    p = Publication(tmp_path / "bundle")
    source = p.put_source("candidate", f["bundle"].source(f["baseline"]))
    p.commit([source])
    (p.root / source["path"]).parent.joinpath("files/solution.json").write_bytes(b"{}")
    with pytest.raises(ProtocolError, match="extra files"):
        Bundle(p.root).source(source)
    workspace = tmp_path / "native"
    workspace.mkdir()
    (workspace / "kernel.py").write_bytes(b"pass\n")
    with pytest.raises(ProtocolError, match="public C"):
        freeze_native_source(Publication(tmp_path / "native-bundle"), workspace,
                             GitCandidateWorkspace(), expected=f["baseline"])


def test_v0_never_uses_starter_and_native_abba_separate(native_task, tmp_path):
    f, _, _, workspace = native_task
    execution, evaluator, request = configured(native_task, tmp_path)
    with pytest.raises(UnsupportedExecution, match="ABBA"):
        evaluator.prepare(replace(request, mode="abba"))
    assert not execution.root.exists()
    (workspace / "kernel.py").write_bytes(f["bundle"].source(f["starter"]))
    with pytest.raises(ProtocolError, match="V0"):
        evaluator.prepare(replace(request, mode="v0"))
    assert not (tmp_path / "jobs/one/intent.json").exists()


def test_native_request_routing_and_owner_resolver(native_task, tmp_path):
    from aka.evaluation.tool_requests import request_from_argv
    f, _, problem, workspace = native_task
    execution, evaluator, _ = configured(native_task, tmp_path)
    operation = replace(evaluator.operation, mode="v0", expected_source=f["baseline"])
    resolved = []
    def owner(request):
        resolved.append(request)
        return operation
    factory = SchemaEvaluatorFactory(execution=execution, operation=owner)
    config = EvaluationConfiguration(timeout=60)
    command = factory.create_command(config, evaluation_input=problem.evaluation)
    params = dict(workspace=workspace, evaluation_input=problem.evaluation, configuration=config,
                  workspace_provider=GitCandidateWorkspace())
    selected = request_from_argv(["--kind", "run", "--no-sync"], **params)
    first = command.prepare(selected.request)
    assert first.context["mode"] == "v0"
    assert first.context["source_binding"]["source"]["sha256"] == f["baseline"]["sha256"]
    assert command.prepare(selected.request).steps[0].submission == first.steps[0].submission
    assert len(resolved) == 2 and not (tmp_path / "jobs/one/intent.json").exists()
    for argv in (["--shape-id", "shape_m1_n128"], ["--baseline-path", "baseline.py"],
                 ["--mode", "correctness_only"], ["--timed-runs", "1"], ["--multi-seed", "5"]):
        with pytest.raises(UnsupportedExecution, match="owns all cases"):
            request_from_argv(argv, **params)


@pytest.mark.parametrize("replacement,failed_check", [
    (None, None),
    (b"out[:, :-1].copy_(torch.softmax(x, dim=-1)[:, :-1])", "out_fully_written"),
    (b"x.zero_(); out.copy_(torch.softmax(x, dim=-1))", "x_bytes"),
    (b"out.fill_(0.5)", "elementwise_tolerance"),
    (b"out.set_(torch.softmax(x, dim=-1))", "out_storage"),
])
def test_full_task_through_real_aka_interfaces(native_task, tmp_path, replacement, failed_check):
    f, _, _, workspace = native_task
    if replacement:
        data = (workspace / "kernel.py").read_bytes().replace(b"out.copy_(torch.softmax(x, dim=-1))", replacement)
        (workspace / "kernel.py").write_bytes(data)
    execution, evaluator, request = configured(native_task, tmp_path)
    if replacement is None:
        request = replace(request, mode="v0")
    validate_executor(execution)
    prepared = evaluator.prepare(request)
    assert prepared.inputs == () and prepared.steps[0].prepared_inputs is None
    # Full native Experiment calls prepare/run/observe/interpret with real objects.
    result = LegacyExperiment(execution, evaluator).run(request)
    facts = execution.channel.executor.query("one")
    worker = facts["worker"]
    assert worker["loaded"]["kernel.py"] == sha256((workspace / "kernel.py").read_bytes())
    assert worker["source"]["sha256"] == prepared.context["source_binding"]["source"]["sha256"]
    assert worker["permissions"]["seccomp_tsync"]
    correct = worker["result"]["correctness"]
    assert len(correct["basic_calls"]) == 72
    assert len(correct["state_reuse"][0]["calls"]) == 2
    if failed_check:
        assert result.gate == "FAIL"
        assert any(not c["checks"][failed_check] for c in correct["basic_calls"])
        if failed_check == "out_storage":
            assert all(c["checks"]["return_identity"] for c in correct["basic_calls"])
    else:
        assert correct["passed"]
        assert result.gate == "UNAVAILABLE" and not result.passed
        assert len(worker["result"]["timing"]["batches"]) == 80
        assert worker["result"]["timing"]["complete"]
        report = json.loads(next(a.content for a in result.artifacts if a.name == "schema-evaluation.json"))
        assert report["correctness"]["status"] == report["performance"]["status"] == "unknown"
        assert report["extensions"]["information"]["qualification"] == "none"
    assert sorted(x.name for x in workspace.iterdir()) == ["acceptance.md", "contract.yaml", "kernel.py"]
    # Original native controller and receipt are reused; no fresh evaluation.
    controller = (tmp_path / "jobs/one/controller.json").read_bytes()
    replay = LegacyExperiment(execution, evaluator).run(request)
    assert replay.gate == result.gate
    assert (tmp_path / "jobs/one/controller.json").read_bytes() == controller
    if replacement is None:
        from aka.evaluation.tool_requests import request_from_argv
        config = EvaluationConfiguration(timeout=60)
        selected = request_from_argv(["--kind", "run", "--version", "v0", "--no-sync"], workspace=workspace,
            evaluation_input=native_task[2].evaluation, configuration=config, workspace_provider=GitCandidateWorkspace())
        factory = SchemaEvaluatorFactory(execution=execution, operation=evaluator.operation)
        command_evaluator = factory.create_command(config, evaluation_input=native_task[2].evaluation)
        result = LegacyExperiment(execution, command_evaluator=command_evaluator).run_command(selected.request)
        payload = json.loads(result.stdout.split("RESULT_JSON=", 1)[1])
        assert payload["all_pass"] is False and payload["comparison"] is None
        assert "random_s17" not in result.stdout and "frozen_inputs" not in result.stdout
        assert (tmp_path / "jobs/one/controller.json").read_bytes() == controller


def test_subject_report_and_loaded_byte_mismatch(native_task, tmp_path):
    # Also exercise actual worker compilation of CRLF + non-ASCII + no final LF.
    workspace = native_task[3]
    data = (workspace / "kernel.py").read_bytes().replace(b"\n", b"\r\n") + "# 原件".encode()
    (workspace / "kernel.py").write_bytes(data)
    execution, evaluator, request = configured(native_task, tmp_path)
    prepared = evaluator.prepare(request)
    native_result = execution.run(prepared.steps[0])
    original = execution.consume(prepared.context, native_result)
    report, worker = original["report"], original["worker"]
    assert worker["loaded"]["kernel.py"] == sha256(data)
    source_binding = bind_source(original["bundle"], original["source_binding"]["source"], GitCandidateWorkspace())
    applicable = dict(task_bundle=original["bundle"], task=prepared.context["task"], source_binding=source_binding,
                      channel=original["channel"], environment=worker["environment"],
                      frozen_inputs_sha256=worker["frozen_inputs_sha256"])
    assert require_applicable(original, **applicable) == original["report_ref"]
    with pytest.raises(ProtocolError, match="not applicable"):
        require_applicable(original, **{**applicable, "channel": {"other": "profile"}})
    with pytest.raises(ProtocolError, match="not applicable"):
        require_applicable(original, **{**applicable, "frozen_inputs_sha256": "sha256:" + "0" * 64})
    forged_transport = replace(native_result, stdout=jcs({"report": "pass"}).decode())
    with pytest.raises(ProtocolError, match="original Schema receipt"):
        execution.consume(prepared.context, forged_transport)
    forged_pass = {**report, "correctness": {**report["correctness"], "status": "pass"}}
    with pytest.raises(ProtocolError, match="CUDA evidence"):
        verify_report_binding(original["bundle"], forged_pass, prepared.context, worker)
    # Pure projection branch exercise with synthetic facts, not a GPU receipt:
    # all original-GPU checks may pass without any R0 fidelity or ABBA claim.
    hypothetical = {**report, **{name: {**report[name], "status": "pass"}
                                  for name in ("build", "correctness", "performance")}, "extensions": {}}
    projected = native_verification({**original, "report": hypothetical,
        "worker": {**worker, "environment": {**worker["environment"], "mode": "cuda"}}})
    assert projected.passed and projected.improvement_pct is None and projected.incumbent_latency_us is None
    assert len(projected.candidate_latency_us_by_shape) == 4
    for key in ("source_sha256", "evaluator_sha256", "inputs_sha256", "timing_sha256", "task_sha256"):
        bad = {**report, "subject": {**report["subject"], key: "sha256:" + "0" * 64}}
        with pytest.raises(ProtocolError, match="subjects"):
            verify_report_binding(original["bundle"], bad, prepared.context, worker)
    wrong_worker = {**worker, "loaded": {**worker["loaded"], "kernel.py": "sha256:" + "0" * 64}}
    with pytest.raises(ProtocolError, match="actual loaded"):
        verify_report_binding(original["bundle"], report, prepared.context, wrong_worker)
    with pytest.raises(ProtocolError, match="environment"):
        verify_report_binding(original["bundle"], report, prepared.context, {**worker, "environment": {}})
    with pytest.raises(ProtocolError, match="frozen input"):
        verify_report_binding(original["bundle"], report, prepared.context, {**worker, "frozen_inputs_sha256": "sha256:" + "0" * 64})
    changed = {**prepared.context, "source_binding": {**prepared.context["source_binding"], "kernel_sha256": "sha256:" + "0" * 64}}
    with pytest.raises(ProtocolError, match="source/channel"):
        execution.consume(changed, native_result)
    path = request.workspace / "kernel.py"
    path.write_bytes(path.read_bytes() + b"# changed\n")
    assert execution.read_operation("one")["report_ref"] == original["report_ref"]
    with pytest.raises((ProtocolError, FileExistsError)):
        evaluator.prepare(request)
