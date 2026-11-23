"""Retained validation reproduces the existing module Skill fingerprint order."""
import pytest

from aka.contracts.agent_resources import AgentSkill
from aka.resources.plugins import PluginRegistry
from aka.resources.skills import fingerprint
from aka.resources.snapshots import inspect_snapshot, restore_snapshot


def test_retained_skill_uses_original_path_order_and_rejects_changed_files(tmp_path):
    source = tmp_path / "skill"
    (source / "assets/cuda").mkdir(parents=True)
    (source / "assets/cuda-cutlass").mkdir()
    (source / "SKILL.md").write_text("# Synthetic CPU skill\n")
    first = source / "assets/cuda/binding.cpp"
    first.write_text("// retained content\n")
    (source / "assets/cuda-cutlass/kernel.cu").write_text("// second retained content\n")
    paths = [path.relative_to(source) for path in source.rglob("*") if path.is_file()]
    for path in paths:
        (source / path).chmod(0o600)
    assert sorted(paths) != sorted(paths, key=lambda path: path.as_posix())
    skill = AgentSkill("cpu-order", source, "CPU fixture")
    original_fingerprint = fingerprint(skill)
    assert original_fingerprint == "eca023f634a2896750d417ca055ab1788043a9801cda033587099fc76a8cf830"
    registry = PluginRegistry(plugin_dir="", plugins=(), resources=(), agent_skills=(skill,))
    expected = registry.resource_snapshot()
    snapshot = registry.export_snapshot(tmp_path / "snapshot")
    assert inspect_snapshot(snapshot).selection_lock == expected
    original = first.read_bytes()
    first.write_text("// changed content\n")
    with pytest.raises(ValueError, match="original resource differs"):
        restore_snapshot(snapshot)
    assert first.read_bytes() != original
    # Restore only this private fixture's missing source root from retained
    # bytes; restoration must retain its original fingerprint and selection.
    import shutil
    shutil.rmtree(source)
    restored = restore_snapshot(snapshot)
    assert first.read_bytes() == original
    assert fingerprint(skill) == original_fingerprint
    assert restored.resource_snapshot() == expected
