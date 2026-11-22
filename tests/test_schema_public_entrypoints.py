"""Real public composition with CPU Agent/native execution fixtures only."""
from dataclasses import replace
import json
from pathlib import Path
import os
import subprocess
import sys

import pytest

from aka.contracts.startup import Invocation
from aka.contracts.workflow_launch import encode_launch
from schema_public_fixture import install_native_cpu, launch, native_selection, schema_config


@pytest.mark.parametrize("paired", [False, True], ids=["design-first", "optimize-first"])
def test_host_selection_constructs_real_controller_and_native_episodes(tmp_path, monkeypatch, paired):
    from aka.bootstrap.host import run_selection
    from aka.execution.legacy.executor import LegacyExecutor
    from aka.workflow.schema.workflow import SchemaWorkflow
    harness = install_native_cpu(monkeypatch, tmp_path)
    monkeypatch.setattr("aka.workflow.schema.workflow.extract",
                        lambda *a, **kw: pytest.fail("these public inputs must not extract"))
    selection = native_selection(harness, paired=paired)
    # The actual service provided by Execution must receive this public setting.
    row = selection.composition.row("execution")
    composition = replace(selection.composition, rows=tuple(
        replace(item, config={**item.config, "measurement_parallelism": 1}) if item.id == row.id else item
        for item in selection.composition.rows))
    selection = replace(selection, composition=composition)
    assert run_selection(selection, Invocation(encode_launch(launch(harness))),
                         schema=schema_config(harness, paired=paired)) == 0
    campaign, = harness.campaigns
    assert isinstance(campaign.driver, SchemaWorkflow)
    assert campaign.driver.planner.agent is campaign.agent
    assert campaign.driver.executor is campaign.tool_runtime.workflow.executor
    assert isinstance(campaign.driver.executor, LegacyExecutor)
    assert campaign.tool_runtime.workflow.measurement_parallelism == 1
    assert [item["kind"] for item in harness.calls] == ["optimize" if paired else "design", "implement", "implement"]
    assert campaign.setup_count == campaign.finish_count == 1
    assert campaign.driver.state.data["counts"]["settled_episodes"] == 2
    assert campaign.driver.state.data["counts"]["agent_calls"] == 3
    assert campaign.driver.state.data["counts"]["tokens"] == 21
    assert campaign.driver.state.data["finalize"]["reason"] == "budget: episodes"


def test_continuous_public_entry_crosses_old_episode_limit_and_honors_explicit_budget(tmp_path, monkeypatch):
    from aka.bootstrap.host import run_selection
    harness = install_native_cpu(monkeypatch, tmp_path)
    config = schema_config(harness, episodes=36)
    assert run_selection(native_selection(harness), Invocation(encode_launch(launch(harness))), schema=config) == 0
    campaign, = harness.campaigns
    assert campaign.driver.state.data["counts"]["settled_episodes"] == 36
    assert len([row for row in harness.calls if row["kind"] == "implement"]) == 36
    assert len([row for row in harness.calls if row["kind"] in {"design", "optimize"}]) == 9
    assert campaign.setup_count == 1
    assert campaign.max_iters == 0


def test_continuous_public_entry_preserves_user_native_budget(tmp_path, monkeypatch):
    from aka.bootstrap.host import run_selection
    harness = install_native_cpu(monkeypatch, tmp_path)
    config = schema_config(harness, episodes=9)
    assert run_selection(native_selection(harness), Invocation(encode_launch(launch(harness, max_iters=2))),
                         schema=config) == 0
    campaign, = harness.campaigns
    assert campaign.max_iters == 2
    assert len([row for row in harness.calls if row["kind"] == "implement"]) <= 2
    assert campaign.driver.state.data["counts"]["settled_episodes"] <= 2


@pytest.mark.parametrize("paired", [False, True], ids=["design-first", "optimize-first"])
def test_workflow_cli_assembles_controller_without_mocking_host(tmp_path, monkeypatch, paired):
    from aka.bootstrap import workflow_cli
    harness = install_native_cpu(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_MODEL", "qwen3.8-max[1m]")
    configuration = tmp_path / "schema.json"
    configuration.write_text(json.dumps(schema_config(harness, paired=paired, episodes=1)))
    patch = tmp_path / "native.patch.json"
    patches = [
        {"id": "resources", "config": {"plugins": [], "bundles": []}},
        {"id": "workflow", "config": {"startup": True, "search": {}, "inherit_state": False,
                                      "output_limit_recovery_retries": 0, "episode_timeout_seconds": 60}}]
    if paired:
        patches.append({"id": "source", "config": {"kind": "kernel",
                        "kernel_demo": str(harness.directory / "input.py")}})
    patch.write_text(json.dumps({"api_version": 1, "patches": patches}))
    # The public CLI resolves its real installed profile. Empty public resource
    # directories suffice because this fixture requests no external tools.
    (tmp_path / "plugins/gpu-wiki").mkdir(parents=True)
    (tmp_path / "gpu-wiki").mkdir()
    arguments = ["--repo-root", str(tmp_path), "--profile", "light", "--patch", str(patch),
        "--problem", str(harness.task), "--workspace-base", str(tmp_path / "runs"),
        "--workspace-suffix", "cli_cpu", "--platform", "cpu", "--framework", "triton",
        "--max-iters", "0", "--target-util", "0", "--fast-episodes", "0", "--handoff-resumes", "0",
        "--hardware", "L20D", "--schema", "--schema-config", str(configuration)]
    # Wheels register the same public function under the actual `aka` dispatcher.
    from importlib.metadata import entry_points
    entries = entry_points(group="aka.commands")
    if any(entry.name == "workflow" for entry in entries):
        from aka.cli import main
        assert main(["workflow", *arguments]) == 0
    else:
        assert workflow_cli.main(arguments) == 0
    assert [row["kind"] for row in harness.calls] == ["optimize" if paired else "design", "implement"]
    assert type(harness.campaigns[0].driver).__name__ == "SchemaWorkflow"


@pytest.mark.parametrize("entry", ["host", "cli"])
def test_public_off_entry_does_not_import_parser_or_read_schema_assets(tmp_path, entry):
    script = Path(__file__).with_name("schema_public_off_probe.py")
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    result = subprocess.run([sys.executable, str(script), str(tmp_path), entry], env=env,
                            text=True, capture_output=True, check=True)
    assert json.loads(result.stdout.splitlines()[-1])["episodes"] == 2
