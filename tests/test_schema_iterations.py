"""Automatic Schema entry wiring; no model or GPU calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from aka.bootstrap import workflow_cli


def test_cli_forwards_original_native_request_to_shared_loop(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setitem(sys.modules, "kernelschema", SimpleNamespace(api=SimpleNamespace(
        iterate=lambda **kw: seen.append(kw) or {"status": "completed", "stop_reason": "keep"})))
    base = ["--repo-root", str(tmp_path), "--problem", str(tmp_path / "problem"),
            "--platform", "L20D", "--framework", "Triton", "--max-iters", "7",
            "--shape-batch-size", "1", "--workspace-base", str(tmp_path / "native"),
            "--patch", str(tmp_path / "native-patch.json"), "--schema", "--schema-iterate",
            "--schema-task", str(tmp_path / "task.md"), "--schema-out", str(tmp_path / "stages"),
            "--schema-rounds", "3", "--schema-runtime-profile", str(tmp_path / "writer.json")]
    assert workflow_cli.main(base) == 0
    options = seen[0]
    launch = options["_launch"]
    assert launch.request.settings.max_iters == 7 and launch.request.runtime.shape_batch_size == 1
    assert set(launch.provided_settings) == {"platform", "framework", "max_iters"}
    assert launch.provided_runtime == ("shape_batch_size",)
    assert launch.request.problem.snapshot_directory is not None
    assert options["schema"] is None and options["rounds"] == 3
    assert options["owner_account"] is None
    assert options["patches"] == [tmp_path / "native-patch.json"]
    assert options["runtime_profile"] == tmp_path / "writer.json"
    assert workflow_cli.main([*base, "--schema-path", str(tmp_path / "root.yaml")]) == 0
    assert seen[1]["schema"] == tmp_path / "root.yaml"


def test_explicit_writing_options_require_iteration_and_task(tmp_path):
    base = ["--repo-root", str(tmp_path), "--problem", str(tmp_path), "--platform", "L20D", "--schema"]
    with pytest.raises(SystemExit, match="2"):
        workflow_cli.main([*base, "--schema-iterate"])
    with pytest.raises(SystemExit, match="2"):
        workflow_cli.main([*base, "--schema-path", "root.yaml", "--schema-rounds", "2"])
    with pytest.raises(SystemExit, match="2"):
        workflow_cli.main([*base, "--schema-iterate", "--schema-config", "legacy.json"])


def test_cli_off_never_imports_optional_code_even_with_writing_flags(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    script = '''
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == "kernelschema" or fullname.startswith(("kernelschema.", "aka.workflow.schema")):
            raise AssertionError("optional import while off: " + fullname)
sys.meta_path.insert(0, BlockOptional())
from aka.bootstrap import workflow_cli
workflow_cli.run_profile = lambda *args, **kwargs: 0
base = ["--repo-root", sys.argv[1], "--problem", sys.argv[1], "--platform", "L20D",
        "--schema-iterate", "--schema-task", "/absent/task", "--schema-out", "/absent/run"]
assert workflow_cli.main(base) == 0
assert workflow_cli.main(base + ["--no-schema"]) == 0
'''
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True, capture_output=True,
                   env={**os.environ, "PYTHONPATH": str(repo / "src")})


def test_standard_retains_native_launch_fields_in_each_fresh_workspace(tmp_path):
    from aka.contracts.workflow import CampaignRuntimeSettings, WorkflowSettings
    from aka.contracts.workflow_launch import decode_launch
    from aka.workflow.legacy.requests import create_launch
    from aka.workflow.schema.standard import implement
    from test_schema_standard import design, task
    problem = task(tmp_path)
    original = create_launch(problem, workspace_base=tmp_path / "selected",
        settings=WorkflowSettings(platform="L20D", framework="Triton", max_iters=9),
        runtime=CampaignRuntimeSettings(shape_batch_size=1),
        provided_settings=("platform", "framework", "max_iters"), provided_runtime=("shape_batch_size",))
    result = implement(schema=design(tmp_path), problem=problem, platform="L20D",
        repo_root=Path(__file__).resolve().parents[1], workspace=tmp_path / "new-version",
        _launch=original, prepare_only=True)
    actual = decode_launch(result["invocation"])
    assert actual.request.settings == original.request.settings
    assert actual.request.runtime == original.request.runtime
    assert actual.provided_settings == original.provided_settings
    assert actual.provided_runtime == original.provided_runtime
    assert actual.request.problem == original.request.problem
    assert actual.workspace_base == original.workspace_base
    assert actual.request.workspace == tmp_path / "new-version" and actual.exact_workspace
    assert not actual.request.workspace.exists()


@pytest.mark.parametrize("exit_code,expected", [(75, "blocked"), (130, "interrupted"), (143, "interrupted")])
def test_native_environment_or_interruption_exit_cannot_advance_writer(tmp_path, monkeypatch, exit_code, expected):
    from aka.bootstrap import host
    from aka.workflow.schema.standard import implement
    from test_schema_standard import design, task
    def run(selection, invocation):
        (tmp_path / "native").mkdir()
        return exit_code
    monkeypatch.setattr(host, "run_selection", run)
    result = implement(schema=design(tmp_path), problem=task(tmp_path), platform="L20D",
        repo_root=Path(__file__).resolve().parents[1], workspace=tmp_path / "native")
    assert result["status"] == expected and result["exit_code"] == exit_code
    assert json.loads((tmp_path / "native/schema-result.json").read_text()) == result


@pytest.mark.parametrize("outcome", ["missing_model_receipt", "pending_gpu", "settled_failure"])
def test_original_native_budget_controls_next_writer(tmp_path, monkeypatch, outcome):
    from aka.bootstrap import host
    from aka.contracts.agent import TokenUsage
    from aka.contracts.execution import JobResult
    from aka.contracts.workflow_launch import decode_launch
    from aka.workflow.schema.standard_budget import StandardBudget
    from kernelschema import api
    from kernelschema.budget import OwnerAccount
    from test_schema_standard import design, task
    limits = {"calls": {"coder": 1, "planner": 0, "reviewer": 0, "extract": 0},
              "model_calls": 1, "evaluator_jobs": 1, "active_ms": 60000, "tokens": None}
    account = OwnerAccount.create(tmp_path / "synthetic-owner.json", owner="cpu-fixture", limits=limits)
    writer_calls, retained = [], {}
    def fail():
        raise RuntimeError("synthetic native failure; original receipt state retained")
    def native(selection, invocation):
        config = selection.composition.row("workflow").config
        workspace = decode_launch(invocation.argv).request.workspace
        workspace.mkdir()
        budget = StandardBudget(config["owner_account"], workspace=workspace,
            identity=config["schema_context"]["identity"], limits=config["owner_limits"])
        try:
            if outcome == "missing_model_receipt":
                budget.model(lambda reservation: fail())
            elif outcome == "pending_gpu":
                budget.execution(budget.take("evaluator"), JobResult(1))
            else:
                budget.model(lambda reservation: SimpleNamespace(session_id="synthetic", exit_status=1,
                    timed_out=False, terminal_usage=TokenUsage(4, 3, 0, 0, 7, "exact"), observation_errors=()))
            fail()
        finally:
            budget.close()
            retained["account"] = account.path.read_bytes()
            retained["receipts"] = {path: path.read_bytes() for path in budget.directory.glob("*.json")}
    def optimize(**options):
        writer_calls.append(options)
        return {"status": "completed", "decision": "keep"}
    monkeypatch.setattr(host, "run_selection", native)
    monkeypatch.setattr(api, "optimize", optimize)
    task_file = tmp_path / "task.md"
    task_file.write_text("CPU fixture; no external calls\n")
    result = api.iterate(task=task_file, schema=design(tmp_path), problem=task(tmp_path),
        repo_root=Path(__file__).resolve().parents[1], platform="L20D", out=tmp_path / "iteration",
        owner_account=account.path, owner_limits={"model_calls": 1, "evaluator_jobs": 1})
    original = json.loads(Path(result["stages"][0]["result_file"]).read_text())
    unsettled = outcome != "settled_failure"
    assert original["status"] == "failed" and original["error"]["type"] == "RuntimeError"
    assert original["error"]["message"] == "synthetic native failure; original receipt state retained"
    assert bool(original["usage"]["unsettled"]) is unsettled
    assert len(writer_calls) == int(not unsettled)
    assert result["status"] == ("blocked" if unsettled else "failed")
    assert account.path.read_bytes() == retained["account"]
    assert all(path.read_bytes() == data for path, data in retained["receipts"].items())


def test_stage_usage_excludes_other_groups_in_same_parent_request(tmp_path):
    from aka.contracts.agent import TokenUsage
    from aka.workflow.schema.standard_budget import StandardBudget
    from kernelschema.budget import OwnerAccount
    limits = {"calls": {"coder": 2, "planner": 0, "reviewer": 0, "extract": 0},
              "model_calls": 2, "evaluator_jobs": 2, "active_ms": 60000, "tokens": None}
    account = OwnerAccount.create(tmp_path / "synthetic-owner.json", owner="cpu-fixture", limits=limits)
    account.register("parent", "sha256:" + "1" * 64, limits, root=str(tmp_path))
    options = dict(identity="synthetic-schema", limits={"model_calls": 1, "evaluator_jobs": 1}, run_id="parent")
    first = StandardBudget(account.path, workspace=tmp_path / "first", **options)
    first.model(lambda reservation: SimpleNamespace(session_id="synthetic", exit_status=0, timed_out=False,
        terminal_usage=TokenUsage(4, 3, 0, 0, 7, "exact"), observation_errors=()))
    first.close()
    second = StandardBudget(account.path, workspace=tmp_path / "second", **options)
    reservation = second.take("coder")
    account.mark_unknown(reservation["operation_id"], "synthetic missing receipt")
    second.close()
    before = account.path.read_bytes()
    assert first.unsettled() == []
    assert second.unsettled() == [reservation["operation_id"]]
    assert account.path.read_bytes() == before
