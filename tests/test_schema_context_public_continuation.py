"""Public driver_factory reconnects an existing CPU root at its next E boundary."""
from copy import deepcopy
from pathlib import Path

import pytest

from aka.contracts.startup import Invocation
from aka.contracts.workflow_launch import encode_launch
from aka.workflow.schema._native import TaskConstraintReview
from aka.workflow.schema.execution_context import CORRECTABLE, sha256
from aka.workflow.schema.materials import DesignRef, read_design
from aka.workflow.schema.run_state import RecoveryRequired
from aka.workflow.schema.workflow import SchemaWorkflow
from schema_public_fixture import install_native_cpu, native_selection, launch, schema_config


def test_public_factory_new_operation_corrects_legacy_parent_without_new_root(tmp_path, monkeypatch):
    from aka.bootstrap.host import run_selection
    from aka.workflow.light import plugin as light
    harness = install_native_cpu(monkeypatch, tmp_path)
    source = harness.fixture / "root.yaml"
    corrected = source.read_bytes()
    source.write_bytes(corrected.replace(b"Architecture: sm_103", b"Architecture: sm_89")
        .replace(b"Language: Triton", b"Language: CUDA C++17")
        .replace(b"Toolchain: Triton 3.7.1", b"Toolchain: nvcc CUDA 12"))
    first = True
    new_context, plan = SchemaWorkflow._new_context, SchemaWorkflow._plan
    monkeypatch.setattr(SchemaWorkflow, "_new_context", lambda self, *args:
        None if first else new_context(self, *args))
    def pause_at_e(self, campaign):
        if first and self.state.data["phase"] == "optimize":
            raise RecoveryRequired("synthetic old owner stopped at a settled E boundary")
        return plan(self, campaign)
    monkeypatch.setattr(SchemaWorkflow, "_plan", pause_at_e)
    create = light.create_workflow
    def embedding(*args, driver_factory=None, **kwargs):
        def factory(request):
            driver = driver_factory(request)
            if not first:
                with driver.state.owned():
                    task = driver.state.data["prepared"]["public_task"]
                driver.task_constraint_review = TaskConstraintReview(sha256(task.encode()),
                    {path: {"status": "unspecified", "value": None} for path in CORRECTABLE})
                driver.resume_after_repair("CPU owner supplied exact Task review and native discovery adapter")
            return driver
        return create(*args, driver_factory=factory, **kwargs)
    monkeypatch.setattr(light, "create_workflow", embedding)
    selected = native_selection(harness)
    invocation = Invocation(encode_launch(launch(harness)))
    config = schema_config(harness, episodes=5)
    with pytest.raises(RecoveryRequired, match="settled E boundary"):
        run_selection(selected, invocation, schema=config)
    old_driver = harness.campaigns[0].driver
    old = deepcopy(old_driver.state.data)
    parent = DesignRef.from_dict(old["design"])
    parent_bytes = read_design(parent).checked.files["root.yaml"]
    original_calls = deepcopy(old["calls"])
    original_episodes = deepcopy(old["episodes"])
    root, root_config = old_driver.root_ref, deepcopy(old_driver.state.config)
    assert old["counts"]["episodes"] == 4 and old["counts"]["agent_calls"] == 5
    first = False
    source.write_bytes(corrected)
    assert run_selection(selected, invocation, schema=config) == 0
    new_driver = harness.campaigns[-1].driver
    state = new_driver.state.data
    assert new_driver is not old_driver and new_driver.root_ref == root and new_driver.state.config == root_config
    assert state["prepared"] == old["prepared"]
    assert read_design(parent).checked.files["root.yaml"] == parent_bytes
    assert state["design_version"] == 2
    assert read_design(DesignRef.from_dict(state["design"])).manifest["parent"] == parent.as_dict()
    assert {key: state["calls"][key] for key in original_calls} == original_calls
    assert {key: state["episodes"][key] for key in original_episodes} == original_episodes
    assert state["counts"]["agent_calls"] == 7 and state["counts"]["tokens"] == 49
    assert state["counts"]["episodes"] == 5 and state["counts"]["dispatches"] == old["counts"]["dispatches"]
    frozen = state["planner_request"]["execution_context_sha256"]
    assert state["episodes"]["5"]["binding"]["execution_context_sha256"] == frozen
    assert [call["kind"] for call in harness.calls] == ["design", *(["implement"] * 4), "optimize", "implement"]
