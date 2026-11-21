"""CPU integration of Schema assistance with the retained native Light loop."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aka.bootstrap.workflow_cli import profile_path
from aka.contracts.agent import AgentRunRequest, AgentRunResult, TokenUsage
from aka.workflow.schema.standard import SchemaAgent, collect_result, implement, schema_context, select


def design(tmp_path):
    directory = tmp_path / "schema"
    directory.mkdir()
    (directory / "root.yaml").write_text("DESIGN: one block per row\n")
    (directory / "dispatch.md").write_text("root.yaml handles the selected task\n")
    (directory / "unselected-manual.md").write_text("Do not include an entire manual\n")
    return directory


def task(tmp_path):
    root = tmp_path / "synthetic-bench"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/run_eval.py").write_text("# CPU fixture; never executed\n")
    (root / "src/atrex_bench").mkdir(parents=True)
    (root / "src/atrex_bench/__init__.py").write_text("")
    problem = root / "operators/row"
    problem.mkdir(parents=True)
    (problem / "reference.py").write_text("class Model: pass\n")
    (problem / "input.py").write_text("# CPU fixture; never executed\n")
    (problem / "shapes.json").write_text(json.dumps({"0": {"input_kwargs": {"N": 4}}}))
    return problem


def test_only_workflow_changes_and_context_is_frozen(tmp_path):
    from aka.bootstrap.profile import resolve_profile
    schema = design(tmp_path)
    variables = {"repo_root": str(Path(__file__).resolve().parents[1])}
    plain = resolve_profile(profile_path("light"), variables=variables)
    selected = select(profile_path("light"), schema=schema, variables=variables)
    for row in plain.composition.rows:
        actual = selected.composition.row(row.id)
        if row.id != "workflow":
            assert actual == row
    workflow = selected.composition.row("workflow")
    assert workflow.name == "aka.workflow.schema.standard_plugin"
    context = workflow.config["schema_context"]
    assert {item["name"] for item in context["files"]} == {"root.yaml", "dispatch.md"}
    (schema / "root.yaml").write_text("changed afterwards")
    assert "one block per row" in json.dumps(context)


def test_same_schema_reaches_two_real_light_episodes(tmp_path):
    from schema_cpu_fixture import make_engine
    schema = design(tmp_path)
    campaign, _ = make_engine(tmp_path)
    native = campaign.agent
    campaign.agent = SchemaAgent(native, schema_context(schema))
    campaign.max_iters, campaign.token_budget, campaign.handoff_resumes = 2, 14, 0
    campaign.run()
    assert len(native.calls) == 2
    assert all("one block per row" in call["prompt"] for call in native.calls)
    # The synthetic handoff is invalid, so normal Light retains the incumbent.
    assert native.calls[1]["received"] == native.calls[0]["received"]
    assert campaign.setup_count == campaign.finish_count == 1
    assert len(campaign.evaluations) == 1  # unchanged original synthetic V0


def test_agent_keeps_tools_and_native_requests_and_avoids_resume_duplication(tmp_path):
    calls = []
    native_session = SimpleNamespace(invoke=lambda prompt, **kw: calls.append((prompt, kw)), close=lambda: None)
    native_runtime = SimpleNamespace(run=lambda request: calls.append(request))
    native = SimpleNamespace(open_session=lambda *a, **kw: native_session,
        runtime=lambda _: native_runtime, tool_session=object(), configuration_identity="native", close=lambda: None)
    wrapped = SchemaAgent(native, schema_context(design(tmp_path)))
    assert wrapped.tool_session is native.tool_session
    request = AgentRunRequest(tmp_path, "task constraints", 7, environment={"tool": "same"})
    wrapped.runtime("claude").run(request)
    assert calls[-1].environment is request.environment
    assert calls[-1].prompt.endswith("task constraints")
    session = wrapped.open_session("claude", tmp_path)
    session.invoke("episode", resume=False)
    assert "one block per row" in calls[-1][0]
    session.invoke("repair", resume=True)
    assert calls[-1] == ("repair", {"resume": True})


def test_prepare_only_uses_native_task_without_legacy_gates_or_network(tmp_path, monkeypatch):
    from aka.bootstrap import host
    monkeypatch.setattr(host, "run_selection", lambda *a: pytest.fail("prepare-only executed services"))
    result = implement(schema=design(tmp_path), problem=task(tmp_path),
        repo_root=Path(__file__).resolve().parents[1], workspace=tmp_path / "native", platform="L20D",
        prepare_only=True)
    assert result["status"] == "prepared" and result["exit_code"] is None
    assert not (tmp_path / "native").exists()


def test_feedback_retains_failure_and_performance_without_exit_code_pass(tmp_path):
    records = tmp_path / ".atrex_versions"
    records.mkdir()
    failure = {"version": "v2", "correctness": {"status": "FAIL"},
               "performance": {"latency_us_geomean": 3.25, "measurement_status": "partial"}}
    (records / "v2.json").write_text(json.dumps(failure))
    result = collect_result(tmp_path, exit_code=0, schema_identity="selected")
    assert result["versions"][0]["record"] == failure
    assert result["feedback_files"] == [str(records / "v2.json")]
    assert "correctness" not in result
    assert result["status"] == "completed"


def test_completed_api_call_reuses_exact_result(tmp_path, monkeypatch):
    from aka.bootstrap import host
    workspace = tmp_path / "native"
    calls = []
    def run(selection, invocation):
        calls.append(invocation)
        workspace.mkdir()
        return 0
    monkeypatch.setattr(host, "run_selection", run)
    options = dict(schema=design(tmp_path), problem=task(tmp_path),
        repo_root=Path(__file__).resolve().parents[1], workspace=workspace, platform="L20D")
    first = implement(**options)
    assert implement(**options) == first
    assert len(calls) == 1 and Path(first["artifacts"]["feedback"]).is_file()
    with pytest.raises(ValueError, match="different request"):
        implement(**{**options, "max_iters": 5})


def test_native_failure_returns_recorded_feedback(tmp_path, monkeypatch):
    from aka.bootstrap import host
    workspace = tmp_path / "native"
    def fail(selection, invocation):
        workspace.mkdir()
        raise RuntimeError("synthetic native evaluation failure")
    monkeypatch.setattr(host, "run_selection", fail)
    result = implement(schema=design(tmp_path), problem=task(tmp_path),
        repo_root=Path(__file__).resolve().parents[1], workspace=workspace, platform="L20D")
    assert result["status"] == "failed"
    assert result["error"]["message"] == "synthetic native evaluation failure"
    assert json.loads(Path(result["artifacts"]["feedback"]).read_text()) == result


def test_cli_on_uses_raw_path_and_off_ignores_optional_paths(tmp_path, monkeypatch):
    from aka.bootstrap import workflow_cli
    from aka.workflow.schema import standard
    original, assisted = [], []
    monkeypatch.setattr(workflow_cli, "run_profile", lambda *a, **kw: original.append((a, kw)) or 0)
    monkeypatch.setattr(standard, "run", lambda *a, **kw: assisted.append((a, kw)) or 0)
    argv = ["--repo-root", str(tmp_path), "--problem", str(tmp_path), "--platform", "L20D"]
    assert workflow_cli.main([*argv, "--no-schema", "--schema-path", "/absent-design"]) == 0
    assert len(original) == 1 and not assisted
    assert workflow_cli.main([*argv, "--schema", "--schema-path", "/selected-design"]) == 0
    assert len(assisted) == 1 and assisted[0][1]["schema"] == Path("/selected-design")


def test_real_composition_boots_native_providers_with_schema(tmp_path):
    from aka.core.boot import boot
    from aka.bootstrap.runtime import selection_runtime
    from aka.bootstrap.continuation import capture
    from aka.workflow.schema.standard_plugin import TrackedWorkflow
    selected = select(profile_path("light"), schema=design(tmp_path),
        variables={"repo_root": str(Path(__file__).resolve().parents[1])})
    with selection_runtime(selected.environment, selected.python_paths):
        selected, _ = capture(selected)
        report = boot(selected.composition, tokens=selected.tokens, required_services=selected.required_services)
        try:
            report.verify_composition()
            workflow = report.service("workflow")
            assert isinstance(workflow, TrackedWorkflow)
            assert isinstance(workflow.native.agent, SchemaAgent)
            assert workflow.native.executor is report.service("executor")
            assert workflow.native.evaluator_factory is report.service("evaluator_factory")
            assert workflow.native.problem is report.service("problem")
        finally:
            report.dispose()


def test_optional_parent_accounts_native_model_and_nested_gpu_jobs(tmp_path):
    from kernelschema.budget import OwnerAccount
    from aka.contracts.execution import CommandSpec, ExecutionFacts, JobResult, ResourceIdentity
    from aka.contracts.execution_recovery import JobReference
    from aka.workflow.schema.standard_budget import StandardBudget, BudgetExecutor
    limits = {"calls": {"coder": 2, "planner": 1, "reviewer": 0, "extract": 0},
              "model_calls": 3, "evaluator_jobs": 2, "active_ms": 60000, "tokens": None}
    account = OwnerAccount.create(tmp_path / "synthetic-owner.json", owner="synthetic", limits=limits)
    budget = StandardBudget(account.path, workspace=tmp_path / "native", identity="synthetic-schema",
                            limits={"model_calls": 2, "evaluator_jobs": 2})
    job = JobResult(0, exit_code=0, resource=ResourceIdentity("gateway", "job-synthetic"),
                    facts=ExecutionFacts(disposition="terminal", remote_terminated=True),
                    job_references=(JobReference("gateway", "job-synthetic"),))
    executor = BudgetExecutor(SimpleNamespace(run=lambda command: job), budget)
    result = SimpleNamespace(session_id="synthetic", exit_status=0, timed_out=False,
                             terminal_usage=TokenUsage(4, 3, 0, 0, 7, "exact"), observation_errors=())
    def invoke(reservation):
        assert executor.run(CommandSpec(tmp_path, ("test",))) is job
        return result
    assert budget.model(invoke) is result
    budget.close()
    operations = list(account.operations(budget.run_id).values())
    actual = [op["latest"] for op in operations if op["latest"]]
    assert sum(row["calls"] for row in actual) == 1
    assert sum(row["evaluator_jobs"] for row in actual) == 1
    assert sum(row["tokens"] for row in actual) == 7
    assert len([op for op in operations if op["released"]]) == 2
    assert not account.fences()


def test_unknown_gpu_preserves_parent_fence(tmp_path):
    from kernelschema.budget import OwnerAccount, BudgetDenied
    from aka.contracts.execution import CommandSpec, JobResult
    from aka.workflow.schema.standard_budget import StandardBudget, BudgetExecutor
    limits = {"calls": {"coder": 1, "planner": 0, "reviewer": 0, "extract": 0},
              "model_calls": 1, "evaluator_jobs": 2, "active_ms": 60000, "tokens": None}
    account = OwnerAccount.create(tmp_path / "synthetic-owner.json", owner="synthetic", limits=limits)
    budget = StandardBudget(account.path, workspace=tmp_path / "native", identity="synthetic-schema",
                            limits={"model_calls": 1, "evaluator_jobs": 2})
    executor = BudgetExecutor(SimpleNamespace(run=lambda command: JobResult(1)), budget)
    executor.run(CommandSpec(tmp_path, ("test",)))
    with pytest.raises(BudgetDenied):
        executor.run(CommandSpec(tmp_path, ("test",)))
    budget.close()
    assert len(account.fences()) == 1
