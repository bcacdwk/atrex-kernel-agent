"""Durable local materials/design release; no model or GPU is involved."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import pytest

from aka.contracts.agent_resources import WorkspaceResourceContribution
from aka.workflow.schema import materials, release
from aka.workflow.schema.language import SchemaError, check_design, collect_design, loads

FIXTURE = Path(__file__).parent / "fixtures/schema_language/minimal"


@pytest.fixture
def frozen(tmp_path):
    source = tmp_path / "public.md"
    source.write_text("Public scalar copy task; private reference is not a role material.\n")
    resource = WorkspaceResourceContribution("public-task", source, "problem/task.md")
    return materials.snapshot_inputs(tmp_path / "retained", (resource,))


def publish(path, frozen, **kwargs):
    return materials.publish_design(path, FIXTURE, version=1, prepared_id="prepared-cpu",
        operation_id="op-1", materials_snapshot=frozen, **kwargs)


def test_material_view_reads_retained_bytes_without_original_sources(tmp_path, frozen):
    (tmp_path / "public.md").unlink()
    view = materials.material_view(frozen)
    assert view == {"problem/task.md": b"Public scalar copy task; private reference is not a role material.\n"}


def test_selection_and_views_have_only_owned_operations_and_explicit_wiki():
    base = materials.select_materials(wiki_ids=[])
    assert len(base) == 2
    assert len(materials.select_materials("unrelated frobnicator input")) == 2
    selected = materials.select_materials(wiki_ids=["mla_expanded_fp8"])
    assert len(selected) > 2
    assert all(".extraction" not in str(item.source) for item in selected)
    for document in (materials.ASSETS / "docs").glob("*.md"):
        text = document.read_text()
        assert "schema_template/" not in text
        assert "schema_extract/" not in text
        assert "check_yaml" not in text
    with pytest.raises(ValueError):
        materials.select_materials(wiki_ids=["absent"])


def test_dynamic_wiki_is_rejected_before_resource_capture(tmp_path):
    with patch("aka.resources.wiki.providers.selected_provider", return_value=object()):
        with pytest.raises(ValueError, match="dynamic Wiki"):
            materials.snapshot_inputs(tmp_path / "retained", ())


def test_runtime_material_is_never_exposed(tmp_path):
    source = tmp_path / "private.txt"
    source.write_text("private evaluator")
    item = WorkspaceResourceContribution("runtime-evaluator", source, visibility="runtime")
    with pytest.raises(ValueError, match="public"):
        materials.snapshot_inputs(tmp_path / "retained", (item,))


def test_manifest_last_readonly_commit_and_matching_replay(tmp_path, frozen):
    destination = tmp_path / "designs/v1"
    ref = publish(destination, frozen)
    artifact = materials.read_design(ref)
    assert artifact.manifest["materials_snapshot"]["identity"] == frozen.identity
    assert artifact.manifest["origins"]["root"]["Math_reshape"]["provider"] == "root"
    assert artifact.manifest["closure_digest"] == materials.closure_digest(collect_design(FIXTURE))
    assert publish(destination, frozen) == ref
    assert all(path.stat().st_mode & 0o222 == 0 for path in (destination / "raw").iterdir())
    assert destination.joinpath("manifest.json").stat().st_mode & 0o222 == 0
    assert "CONTRACT" in loads(artifact.effective_yaml(), "root").value


@pytest.mark.parametrize("phase", ["data", "manifest"])
def test_interrupted_precommit_is_unpublished_and_preserved(tmp_path, frozen, phase):
    directory = tmp_path / "draft"
    original = materials.durable_write_text
    def fail(path, *args, **kwargs):
        if (phase == "data" and path.name == "root.yaml") or (phase == "manifest" and path.name == "manifest.json"):
            raise OSError("synthetic crash before commit")
        return original(path, *args, **kwargs)
    with patch.object(materials, "durable_write_text", side_effect=fail):
        with pytest.raises(OSError):
            publish(directory, frozen)
    assert not (directory / "manifest.json").exists()
    before = {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="no committed manifest"):
        materials.read_design(directory)
    with pytest.raises(ValueError, match="fresh host directory"):
        publish(directory, frozen)
    assert before == {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    # Retained candidate bytes suffice; recovery does not need a new author call.
    assert materials.read_design(publish(tmp_path / "rebuilt", frozen)).manifest["operation_id"] == "op-1"


def test_postcommit_root_registration_gap_reuses_exact_manifest(tmp_path, frozen):
    directory = tmp_path / "committed"
    original = materials.durable_write_text
    def crash_after_commit(path, *args, **kwargs):
        original(path, *args, **kwargs)
        if path.name == "manifest.json":
            raise OSError("crash before root registration")
    with patch.object(materials, "durable_write_text", side_effect=crash_after_commit):
        with pytest.raises(OSError):
            publish(directory, frozen)
    existing = materials.read_design(directory).ref
    assert publish(directory, frozen) == existing


def test_conflicting_operation_and_modified_published_bytes_are_rejected(tmp_path, frozen):
    directory = tmp_path / "committed"
    ref = publish(directory, frozen)
    with pytest.raises(ValueError, match="conflicting"):
        materials.publish_design(directory, FIXTURE, version=1, prepared_id="prepared-cpu",
            operation_id="different-op", materials_snapshot=frozen)
    root = directory / "raw/root.yaml"
    root.chmod(0o600)
    root.write_bytes(root.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="committed closure"):
        materials.read_design(ref)


def test_manifest_identity_and_provenance_are_bound(tmp_path, frozen):
    directory = tmp_path / "committed"
    ref = publish(directory, frozen)
    with pytest.raises(ValueError, match="identity changed"):
        materials.read_design(materials.DesignRef(directory, "0" * 64))
    (directory / "intent.json").write_text("{}")
    with pytest.raises(ValueError, match="durable intent"):
        materials.read_design(ref)


def test_design_does_not_inherit_parent_extraction_approval(tmp_path, frozen):
    with pytest.raises(ValueError, match="inherit"):
        publish(tmp_path / "invalid", frozen, approval_refs=["old-p6"])
    parent = publish(tmp_path / "v1", frozen)
    with pytest.raises(ValueError, match="prepared identity"):
        materials.publish_design(tmp_path / "v2", FIXTURE, version=2, prepared_id="other-task",
            operation_id="op2", materials_snapshot=frozen, parent=parent)


def test_invalid_draft_is_never_published(tmp_path, frozen):
    files = collect_design(FIXTURE)
    files["dispatch.md"] = b"incomplete result"
    directory = tmp_path / "invalid"
    with pytest.raises(SchemaError, match="dispatch.md"):
        materials.publish_design(directory, files, version=1, prepared_id="p", operation_id="o",
                                 materials_snapshot=frozen)
    assert not directory.exists()


def test_local_release_verifies_without_upstream_discovery_or_commands():
    with patch("importlib.util.find_spec", side_effect=AssertionError("discovery forbidden")), patch(
            "subprocess.run", side_effect=AssertionError("commands forbidden")):
        identity = release.verify_release()
    assert identity["language_version"] == "aka.schema.language.v1"
    assert "ks_root" not in identity
    assert not any(name == "kernelschema" or name.startswith("kernelschema.") for name in sys.modules)


def test_release_detects_local_corruption(tmp_path, monkeypatch):
    directory = tmp_path / "schema"
    shutil.copytree(Path(release.__file__).parent, directory, ignore=shutil.ignore_patterns("__pycache__"))
    (directory / "assets/docs/operations.md").write_text("changed rules")
    monkeypatch.setattr(release, "ROOT", directory)
    with pytest.raises(release.SchemaUnavailable, match="changed"):
        release.verify_release()


def test_freezer_consumes_only_local_inventory():
    script = Path(__file__).parents[1] / "tools/freeze_schema_release.py"
    spec = importlib.util.spec_from_file_location("aka_schema_freezer_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with patch("subprocess.check_output", side_effect=AssertionError("Git forbidden")):
        module.freeze(check=True)
    assert all(not name.startswith("kernelschema/") for name in module.inventory()["files"])


def test_business_code_edits_do_not_change_material_release_or_design(tmp_path, frozen, monkeypatch):
    script = Path(__file__).parents[1] / "tools/freeze_schema_release.py"
    spec = importlib.util.spec_from_file_location("aka_schema_freezer_boundary_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / "schema-release"
    shutil.copytree(Path(release.__file__).parent, directory, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(module, "SCHEMA", directory)
    monkeypatch.setattr(release, "ROOT", directory)
    ref = publish(tmp_path / "published", frozen)
    before = module.inventory()
    identity = release.verify_release()["manifest_sha256"]
    assert set(name.split("/")[0] for name in before["files"]) <= {"NOTICE.md", "language", "assets", "prompts"}
    for filename in ("workflow.py", "run_state.py", "planner.py", "materials.py", "extract_stages.py"):
        (directory / filename).write_text("# changed business implementation, not material bytes\n")
    assert module.inventory() == before
    assert release.verify_release()["manifest_sha256"] == identity
    assert materials.read_design(ref).ref == ref
    (directory / "assets/docs/operations.md").write_text("changed material semantics")
    assert module.inventory() != before
    with pytest.raises(release.SchemaUnavailable, match="changed"):
        release.verify_release()


WIKI = json.loads((materials.ASSETS / "wiki-index.json").read_text())["entries"]


@pytest.mark.parametrize("entry", WIKI, ids=[entry["id"] for entry in WIKI])
def test_selected_wiki_bytes_pass_shared_language(entry):
    directory = materials.ASSETS / "ks/schema_wiki" / entry["directory"]
    files = {name: directory.joinpath(name).read_bytes() for name in entry["files"]}
    result = check_design(files, skip_hardware=True)
    assert len(result.resolved.raw_task) == len(entry["files"]) - 1
