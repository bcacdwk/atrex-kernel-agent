"""Check shipped launch recipes statically without importing either GPU input."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from aka.bootstrap.profile import resolve_profile
from aka.bootstrap.workflow_cli import profile_path
from aka.contracts.startup import Invocation
from aka.contracts.workflow import WorkflowSettings
from aka.contracts.workflow_launch import decode_launch, encode_launch
from aka.workflow.legacy.requests import create_launch
from aka.workflow.schema.launch import SchemaConfig, prepare_selection


INPUTS = Path(__file__).resolve().parents[1] / "examples/schema/l20d"


@pytest.mark.parametrize("name", ["vector_add", "fp8_scale_group_gemm"])
def test_public_recipe_resolves_without_dynamic_wiki_checkout(tmp_path, name):
    directory = INPUTS / name
    config_path = directory / "schema-config.json"
    config = SchemaConfig.load(config_path)
    patches = json.loads((INPUTS / "runtime.patch.json").read_text())["patches"]
    if name == "vector_add":
        patches.append({"id": "source", "config": {"kind": "kernel",
                        "kernel_demo": str(directory / "kernel.py")}})
    selection = resolve_profile(profile_path("light-sandbox"), patches=patches,
                                variables={"repo_root": str(INPUTS)})
    launch = create_launch(directory / "benchmark", workspace_base=tmp_path,
        settings=WorkflowSettings(platform="cuda", framework="triton"),
        provided_settings=("platform", "framework"))
    selected, invocation = prepare_selection(selection, Invocation(encode_launch(launch)), config_path)
    assert selected.resources == ()
    assert selected.composition.row("workflow").name == "aka.workflow.schema.plugin"
    assert selected.composition.row("execution").config == {"profile": "legacy", "measurement_parallelism": 1}
    request = decode_launch(invocation.argv).request
    assert request.settings.max_iters == request.settings.convert_after == 0
    assert request.settings.setup_timeout == 3600
    assert request.settings.framework_baseline == "never"
    if name == "vector_add":
        assert config.initial["schema"] == str(directory / "schema")
        assert all(Path(path).is_file() for path in config.initial["evidence"])
        assert config.planner["check_policy"] == {"skip_hardware": True, "style": "advisory"}
    else:
        assert config.initial is None and config.sota is None


def test_supplied_recipe_rejects_reference_default_as_starting_kernel(tmp_path):
    selection = resolve_profile(profile_path("light-sandbox"),
        variables={"repo_root": str(INPUTS)})
    launch = create_launch(INPUTS / "vector_add/benchmark", workspace_base=tmp_path,
        settings=WorkflowSettings(platform="cuda", framework="triton"),
        provided_settings=("platform", "framework"))
    with pytest.raises(ValueError, match="explicitly selected native kernel"):
        prepare_selection(selection, Invocation(encode_launch(launch)), INPUTS / "vector_add/schema-config.json")
