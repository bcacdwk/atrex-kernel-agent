"""Public Resources selection reaches real native drafts without entering Planner inputs."""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json

import pytest

from aka.bootstrap.continuation import capture, restore
from aka.bootstrap.profile import resolve_profile
from aka.bootstrap.runtime import selection_runtime
from aka.bootstrap.workflow_cli import profile_path
from aka.contracts.content import WorkspaceFile
from aka.contracts.problem import AgentProblem, EvaluationInput, InitialSource, LoadedProblem
from aka.contracts.startup import Invocation
from aka.contracts.workflow import WorkflowSettings
from aka.contracts.workflow_launch import decode_launch, encode_launch
from aka.contracts.workspace import RevisionContent
from aka.core.boot import boot
from aka.core.plugin_runtime.schema import PluginError
from aka.resources.plugins import PluginRegistry
from aka.resources.snapshots import inspect_snapshot, restore_snapshot
from aka.workflow.legacy.requests import create_launch
from aka.workflow.light.orchestration.episode_workspace import EpisodeWorkspace
from aka.workflow.schema.launch import prepare_selection
from aka.workflow.schema.materials import material_view, role_material_view


CLIENTS = ("tools/sandbox.py", "tools/plugin.py")
SKILLS = ("gpu-measurement", "runtime-records")


def _plugin(directory):
    directory.mkdir(parents=True)
    (directory / "data").mkdir()
    (directory / "data/guide.txt").write_text("Synthetic CPU resource, no tool invocation.\n")
    (directory / "skill").mkdir()
    (directory / "skill/SKILL.md").write_text("# CPU resource fixture\n")
    (directory / "plugin.json").write_text(json.dumps({
        "api_version": 1, "id": "cpu-resources", "version": "1",
        "tools": {}, "skills": {"cpu-fixture": {"path": "skill"}},
        "resources": {"cpu-guide": "data"}}))
    return directory


@contextmanager
def _campaign(tmp_path, *, schema, selection_kind, late=None):
    stock = _plugin(tmp_path / "plugins/gpu-wiki")
    (tmp_path / "gpu-wiki").mkdir()
    patches = [{"id": "workflow", "config": {"startup": True, "search": {},
                                                "inherit_state": False}}]
    if selection_kind == "custom":
        custom = _plugin(tmp_path / "custom-catalog/cpu")
        patches.append({"id": "resources", "config": {"plugins": [],
            "plugin_dir": str(custom.parent),
            "bundles": ["measurement-session", "research-records", "flydsl-0.3.4"]}})
    elif selection_kind == "empty":
        patches.append({"id": "resources", "config": {"plugins": [], "bundles": []}})
    if late:
        name = "late_resource_" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]
        if late == "resource":
            (tmp_path / "late-data.txt").write_text("Selected after Workflow assembly.\n")
            body = '''from aka.contracts.agent_resources import WorkspaceResourceContribution
def apply(ctx, config):
    value = WorkspaceResourceContribution("late-native", Path(__file__).with_name("late-data.txt"), "reference/late.txt")
    ctx.contribute("workspace_resources", value.name, value)
'''
        else:
            _plugin(tmp_path / "late-provider")
            body = '''from aka.resources.wiki.providers import contribute
class Provider:
    catalog_root = Path(__file__).with_name("late-provider")
    cli_root = None
    def revision(self): return "synthetic-cpu"
    def invoke(self, call): raise AssertionError("no Wiki calls in CPU assembly")
def apply(ctx, config):
    contribute(ctx, Provider())
'''
        (tmp_path / (name + ".py")).write_text(
            f'from pathlib import Path\nname = "{name.replace("_", "-")}"\ninject = ("workflow",)\n' + body)
        patches.append({"id": "late", "insert": {"name": name, "required": True}})
    selected = resolve_profile(profile_path("light"), patches=patches,
                               variables={"repo_root": str(tmp_path)})
    if late:
        selected = replace(selected, python_paths=(tmp_path,), resources=(*selected.resources,
            (tmp_path, (name + ".py", "late-data.txt" if late == "resource" else "late-provider"))))
    request = create_launch(tmp_path / "task", workspace_base=tmp_path / "runs",
        settings=WorkflowSettings(platform="cpu", framework="triton", max_iters=1,
            framework_baseline="never", convert_after=0, setup_timeout=3600))
    invocation = Invocation(encode_launch(request))
    # Capture the ordinary native selection independently of Schema's adapter.
    with selection_runtime(selected.environment, selected.python_paths):
        report = boot(selected.composition, tokens=selected.tokens,
                      required_services=selected.required_services)
        try:
            expected = PluginRegistry().resource_snapshot()
        finally:
            report.dispose()
        if schema:
            selected, invocation = prepare_selection(selected, invocation, {"wiki_ids": []})
        selected, payload = capture(selected)
        assert capture(restore(payload))[1] == payload
        report = boot(selected.composition, tokens=selected.tokens,
                      required_services=selected.required_services)
        try:
            workflow = report.service("workflow")
            launch = decode_launch(invocation.argv)
            problem = LoadedProblem("cpu", tmp_path / "task", tmp_path / "reference.py",
                None, None, "none", agent=AgentProblem(()),
                initial_source=InitialSource("kernel", tmp_path / "kernel.py",
                    (WorkspaceFile("kernel.py", text="def kernel(): return 1\n"),)),
                evaluation=EvaluationInput("synthetic-cpu", tmp_path, (), runtime_root=tmp_path))
            # Real ManagedCampaign.create owns the registry decision here.
            campaign = workflow._campaign(launch.request, problem)
            yield campaign, expected, stock
        finally:
            report.dispose()


def _locks(workspace):
    return {name: (workspace / ".atrex_plugins" / name).read_bytes()
            for name in ("lock.json", "agent-skills.json", "workspace-resources.json")}


@pytest.mark.parametrize("schema", [False, True], ids=["native-off", "schema-on"])
@pytest.mark.parametrize("selection_kind", ["default", "custom", "empty"])
def test_selected_native_resources_reach_episode_and_resume(tmp_path, schema, selection_kind):
    with _campaign(tmp_path, schema=schema, selection_kind=selection_kind) as (campaign, expected, _):
        registry = campaign._module_resources
        assert registry.resource_snapshot() == expected
        owner = campaign.candidate_workspace
        owner.prepare_draft(campaign.workspace)
        owner.install_source(campaign.workspace, RevisionContent({"kernel.py": b"def kernel(): return 1\n"}))
        view = EpisodeWorkspace(campaign.workspace, tmp_path / "episode", resources_registry=registry,
                                workspace_provider=owner, inherit_state=False)
        workspace = view.prepare()
        original = _locks(workspace)
        assert json.loads(original["lock.json"]) == expected["plugins"]
        assert json.loads(original["agent-skills.json"]) == expected["agent_skills"]
        assert json.loads(original["workspace-resources.json"]) == expected["workspace_resources"]
        mounted = (*CLIENTS, *(f"{root}/skills/{name}/SKILL.md"
                    for root in (".agents", ".claude", ".qoder") for name in SKILLS))
        for name in mounted:
            assert (workspace / name).is_file() == (selection_kind != "empty")
        assert (workspace / "reference/dsl/flydsl/0.3.4/guide.md").is_file() == (selection_kind == "custom")
        if selection_kind != "empty":
            # A normal Session restart repairs missing managed links using the
            # same selection, while retaining custom Episode files and locks.
            (workspace / "tools/sandbox.py").unlink()
            (workspace / ".agents/skills/gpu-measurement").unlink()
        (workspace / "tools/local.py").write_text("# private CPU draft utility\n")
        snapshot = registry.export_snapshot(tmp_path / "retained-native")
        restored = restore_snapshot(snapshot)
        assert inspect_snapshot(snapshot).selection_lock == expected
        assert restored.resource_snapshot() == expected
        view.resources_registry = restored
        view.prepare_session()
        assert _locks(workspace) == original
        assert (workspace / "tools/local.py").read_text() == "# private CPU draft utility\n"
        for name in mounted:
            assert (workspace / name).is_file() == (selection_kind != "empty")
        restored.filesystem_access(workspace)
        if schema:
            planner_snapshot = campaign.driver.planner.materials_snapshot
            facts = inspect_snapshot(planner_snapshot)
            assert facts.selection_lock["plugins"]["plugins"] == []
            assert facts.selection_lock["agent_skills"]["skills"] == []
            assert all(name.startswith("schema-materials/") for name in material_view(planner_snapshot))
            assert role_material_view(planner_snapshot)
            assert planner_snapshot.identity != snapshot.identity
            assert restore_snapshot(planner_snapshot).resource_snapshot() == facts.selection_lock
            assert not (workspace / "schema-materials").exists()


def test_old_material_only_locks_are_rejected_without_rewriting(tmp_path):
    with _campaign(tmp_path, schema=True, selection_kind="default") as (campaign, expected, _):
        materials = restore_snapshot(campaign.driver.planner.materials_snapshot)
        workspace = tmp_path / "legacy-episode"
        workspace.mkdir()
        materials.install(workspace)
        original = _locks(workspace)
        with pytest.raises(PluginError) as error:
            campaign._module_resources.install(workspace)
        assert error.value.code == "plugin_changed"
        assert _locks(workspace) == original
        assert campaign._module_resources.resource_snapshot() == expected


@pytest.mark.parametrize("schema", [False, True], ids=["native-off", "schema-on"])
def test_late_selected_contribution_reaches_real_native_draft(tmp_path, schema):
    with _campaign(tmp_path, schema=schema, selection_kind="default", late="resource") as (campaign, expected, _):
        assert campaign._module_resources.resource_snapshot() == expected
        assert "late-native" in {row["name"] for row in expected["workspace_resources"]["resources"]}
        owner = campaign.candidate_workspace
        owner.prepare_draft(campaign.workspace)
        owner.install_source(campaign.workspace, RevisionContent({"kernel.py": b"def kernel(): return 1\n"}))
        view = EpisodeWorkspace(campaign.workspace, tmp_path / "episode",
            resources_registry=campaign._module_resources, workspace_provider=owner)
        workspace = view.prepare()
        assert (workspace / "reference/late.txt").read_text() == "Selected after Workflow assembly.\n"
        for name in (*CLIENTS, *(f".agents/skills/{name}/SKILL.md" for name in SKILLS)):
            assert (workspace / name).is_file()
        campaign._module_resources.filesystem_access(workspace)


def test_late_dynamic_wiki_provider_is_rejected_at_request_binding(tmp_path):
    with pytest.raises(ValueError, match="dynamic Wiki provider"):
        with _campaign(tmp_path, schema=True, selection_kind="empty", late="wiki"):
            pytest.fail("Schema must reject late dynamic Wiki before a native campaign exists")
