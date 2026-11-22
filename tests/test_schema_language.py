"""AKA-owned language regressions: exact values, closure and policy boundaries."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import pytest

from aka.workflow.schema import language as schema
from aka.workflow.schema.language import yaml_source, yaml_format, design_resolution
from aka.workflow.schema.language.diagnostics import encode_value

FIXTURE = Path(__file__).parent / "fixtures/schema_language/minimal"
REGIONS = ("Math_reshape", "Work_schedule", "Tile_pipeline")


@pytest.fixture
def task():
    return {"root": yaml_source.load(FIXTURE / "root.yaml", "root").value}


def add_child(task, name="branch", source="root"):
    task["root"]["META_DATA"]["Dispatch"] = {"when": "All scalar inputs"}
    child = deepcopy(task["root"])
    child.pop("CONTRACT")
    child["META_DATA"]["Dispatch"] = {"id": name, "when": "One selected subdomain"}
    if source:
        child["DESIGN"] = {region: {"from": source} for region in REGIONS}
    task[name] = child
    return child


def snapshot(task):
    files = {f"{name}.yaml": encode_value(value).encode() for name, value in task.items()}
    files["dispatch.md"] = ("## 范围\nSynthetic CPU-only scalar copy\n## 完整设计\n"
        "| 设计 | 具体机制与保留理由 | 关系 |\n| --- | --- | --- |\n" +
        "".join(f"| [{name}]({name}.yaml) | Scalar copy | Current design |\n" for name in task)).encode()
    return files


def test_exact_source_marks_merges_types_and_crlf():
    source = ('base: &base {flag: on, date: 2026-11-02, count: 080}\r\n'
              'copied: {<<: *base, explicit: !!str 1e3, exact: 1.000000000000000000000001}\r\n')
    result = yaml_source.loads(source, "root")
    assert result.source == source
    assert result.source_sha256 == hashlib.sha256(source.encode()).hexdigest()
    assert result.value["copied"] == {"flag": "on", "date": "2026-11-02", "count": 80,
                                       "explicit": "1e3", "exact": Decimal("1.000000000000000000000001")}
    assert result.positions["root.copied.exact"][0] == 2
    assert schema.json_material(result.value)["copied"]["exact"] == {"$decimal": "1.000000000000000000000001"}


@pytest.mark.parametrize("source", ["x: 1\nx: 2", "x: &loop {again: *loop}",
    "x: 1\n---\nx: 2", "x: .inf", "x: .nan", "x: !execute foo", "1: bad", "[]",
    "base: &base {x: 1}\nmerged: {<<: !bad [*base]}"])
def test_yaml_rejects_ambiguous_unsafe_and_nonfinite_values(source):
    with pytest.raises(yaml_source.ReadError):
        yaml_source.loads(source, "root")


def test_parser_node_capacity_is_separate_from_from_resolution(task, monkeypatch):
    monkeypatch.setattr(yaml_source, "MAX_EXPANDED_NODES", 12)
    with pytest.raises(schema.CapacityError):
        yaml_source.loads("a: &x [1,2,3,4]\nb: [*x,*x,*x]", "root")
    monkeypatch.setattr(yaml_source, "MAX_EXPANDED_NODES", 100000)
    monkeypatch.setattr(design_resolution, "MAX_RESOLVED_UNITS", 10)
    with pytest.raises(schema.CapacityError, match="resolved units"):
        schema.check_design(snapshot(task))


def test_file_size_cap_applies_to_memory_and_files(tmp_path, monkeypatch):
    monkeypatch.setattr(yaml_source, "MAX_FILE_BYTES", 8)
    path = tmp_path / "root.yaml"
    path.write_text("x: abcdefg")
    for read in (lambda: yaml_source.loads(path.read_text(), "root"), lambda: yaml_source.load(path, "root")):
        with pytest.raises(schema.CapacityError):
            read()


@pytest.mark.parametrize("case", ["symlink", "hardlink", "parent_link", "extra", "directory", "yml"])
def test_design_collection_rejects_aliases_and_nonclosure(tmp_path, case):
    directory = tmp_path / "design"
    shutil.copytree(FIXTURE, directory)
    if case == "symlink":
        (directory / "root.yaml").unlink()
        (directory / "root.yaml").symlink_to(FIXTURE / "root.yaml")
    elif case == "hardlink":
        (tmp_path / "alias").hardlink_to(directory / "root.yaml")
    elif case == "parent_link":
        (tmp_path / "alias").symlink_to(directory, target_is_directory=True)
        directory = tmp_path / "alias"
    elif case == "extra":
        (directory / "kernel.py").write_text("pass")
    elif case == "directory":
        (directory / "hidden").mkdir()
    else:
        (directory / "root.yaml").rename(directory / "root.yml")
    with pytest.raises((schema.SchemaError, OSError)):
        schema.check_design(directory)


def test_raw_and_provider_closures_are_distinct(task):
    add_child(task, "unused", source=None)
    provider = add_child(task, "provider", source=None)
    provider["DESIGN"]["Work_schedule"] = {"from": "unused"}
    add_child(task, "middle", source="provider")
    child = add_child(task, "child", source="middle")
    child["DESIGN"]["Work_schedule"] = deepcopy(task["root"]["DESIGN"]["Work_schedule"])
    child["DESIGN"]["Tile_pipeline"] = {"from": "provider"}
    child["REALIZATION"] = {"Prefer": [{"Item": "y", "Choice": "Keep scalar copy", "Reason": "Selected behavior"}]}
    result = schema.check_design(snapshot(task)).resolved
    assert result.origins["child"]["Math_reshape"].chain == ("child", "middle", "provider")
    assert set(result.source_manifest("child")) == {"root.yaml", "provider.yaml", "middle.yaml", "child.yaml"}
    assert "unused" not in result.dependency_nodes["child"]
    view = result.effective_document("child")
    assert view["CONTRACT"] == task["root"]["CONTRACT"]
    assert view["META_DATA"] == child["META_DATA"]
    assert view["REALIZATION"] == child["REALIZATION"]
    view["CONTRACT"].clear()
    assert result.raw_task["root"]["CONTRACT"]
    output = schema.effective_yaml(result, "child")
    assert yaml_source.loads(output, "child").value["CONTRACT"] == task["root"]["CONTRACT"]


@pytest.mark.parametrize("case", ["cycle", "root_from", "mixed", "missing", "local_contract", "wrong_id", "shared_name"])
def test_invalid_shared_contract_or_from_graph(task, case):
    child = add_child(task)
    if case == "cycle":
        other = add_child(task, "other")
        child["DESIGN"]["Math_reshape"] = {"from": "other"}
        other["DESIGN"]["Tile_pipeline"] = {"from": "branch"}
    elif case == "root_from":
        task["root"]["DESIGN"]["Math_reshape"] = {"from": "branch"}
    elif case == "mixed":
        child["DESIGN"]["Math_reshape"]["Thesis"] = "Mixed override"
    elif case == "missing":
        child["DESIGN"]["Math_reshape"] = {"from": "absent"}
    elif case == "local_contract":
        child["CONTRACT"] = deepcopy(task["root"]["CONTRACT"])
    elif case == "wrong_id":
        child["META_DATA"]["Dispatch"]["id"] = "other"
    else:
        child["META_DATA"]["Name"] = "different_name"
    with pytest.raises(schema.SchemaError):
        schema.check_design(snapshot(task))


def test_count_limit_includes_root_and_from_consumers(task):
    for index in range(9):
        add_child(task, f"child_{index}")
    assert len(schema.check_design(snapshot(task)).resolved.raw_task) == 10
    add_child(task, "last_child")
    with pytest.raises(schema.CapacityError):
        schema.check_design(snapshot(task))


@pytest.mark.parametrize("case", ["tolerance", "step", "launch", "pipe", "numeric", "unknown_field"])
def test_structural_and_relation_failures_stay_hard_in_advisory_mode(task, case):
    root = task["root"]
    if case == "tolerance":
        root["CONTRACT"]["Algorithm_requirement"]["Golden_reference"]["Tolerance"] = {"absent": "exact"}
    elif case == "step":
        root["DESIGN"]["Tile_pipeline"]["Step"]["NEXT"] = root["DESIGN"]["Tile_pipeline"]["Step"].pop("MAIN")
    elif case == "launch":
        root["DESIGN"]["Work_schedule"]["Flow"]["Launch"].append({"step": ["MAIN"]})
    elif case == "pipe":
        root["DESIGN"]["Tile_pipeline"]["Step"]["MAIN"]["Role"]["MAIN"]["Pipe"] = ["compute:absent"]
    elif case == "numeric":
        root["DESIGN"]["Work_schedule"]["Step"]["MAIN"]["Mapping"]["Worker_count"] = "+"
    else:
        root["DESIGN"]["Math_reshape"]["unknown"] = "Ignored?"
    with pytest.raises(schema.SchemaError):
        schema.check_design(snapshot(task), style="advisory")


def test_style_categories_preserve_numeric_checks_and_pipe_traversal(task):
    root = task["root"]
    root["META_DATA"]["Describe"] = "x" * 151
    role = root["DESIGN"]["Tile_pipeline"]["Step"]["MAIN"]["Role"]["MAIN"]
    # A long but valid local loop bound must still let relation checks run.
    role["Pipe"] = [{"for(i=0:" + "1+" * 51 + "1)": ["compute:y"]}]
    result = schema.check_design(snapshot(task))
    assert any(item.category == "style.length" and ".Pipe" in item.path for item in result.diagnostics)
    assert all(item.severity == "warning" for item in result.diagnostics)
    with pytest.raises(schema.SchemaError) as error:
        schema.check_design(snapshot(task), style="strict")
    assert any(item.category == "style.length" for item in error.value.diagnostics)
    role["Pipe"] = [{"for(i=0:+)": ["compute:y"]}]
    with pytest.raises(schema.SchemaError):
        schema.check_design(snapshot(task))


def test_style_warning_does_not_hide_invalid_structure(task):
    task["root"]["META_DATA"]["Name"] = "A" * 90
    inputs = task["root"]["CONTRACT"]["Algorithm_requirement"]["Input"]
    inputs["INVALID_NAME"] = inputs.pop("x")
    with pytest.raises(schema.SchemaError):
        schema.check_design(snapshot(task))


def test_comment_normalization_refuses_loss_while_advisory_check_retains_source():
    files = schema.collect_design(FIXTURE)
    files["root.yaml"] = b"# preserve author note\n" + files["root.yaml"]
    checked = schema.check_design(files)
    assert any(item.category == "style.comments" for item in checked.diagnostics)
    with pytest.raises(schema.SchemaError):
        schema.normalize_design(files)
    assert checked.files["root.yaml"].startswith(b"# preserve")


def test_normalization_rebinds_shared_symbols_and_preserves_decimal(task):
    root = task["root"]
    algorithm = root["CONTRACT"]["Algorithm_requirement"]
    algorithm["Input"]["X"] = algorithm["Input"].pop("x")
    algorithm["Input"]["X"]["range"] = [Decimal("0.000000000000000000001"), Decimal("1.000000000000000000001")]
    algorithm["Compute"][0]["Math"] = "y = X\n"
    root["DESIGN"]["Math_reshape"]["Rewrite"][0]["Math"] = "y = X\n"
    invariant = root["DESIGN"]["Work_schedule"]["Step"]["MAIN"]["Decomposition"]["Worktile"]["Invariant"]
    invariant["X"] = invariant.pop("x")
    add_child(task)
    files = snapshot(task)
    before = dict(files)
    plan = schema.normalize_design(files)
    assert files == before
    result = dict(files)
    result.update({f"{name}.yaml": document.source.encode() for name, document in plan.documents.items()})
    checked = schema.check_design(result, style="strict")
    assert "x" in checked.resolved.raw_task["root"]["CONTRACT"]["Algorithm_requirement"]["Input"]
    assert plan.corrections
    assert checked.resolved.raw_task["branch"]["DESIGN"]["Math_reshape"] == {"from": "root"}
    assert yaml_source.loads(schema.effective_yaml(checked.resolved, "branch"), "branch").value["CONTRACT"]["Algorithm_requirement"]["Input"]["x"]["range"] == algorithm["Input"]["X"]["range"]


@pytest.mark.parametrize("replace", [b"[absent](missing.yaml)", b"[root](https://example.invalid/root.yaml)", b"[root](../root.yaml)"])
def test_dispatch_requires_real_current_design_links(task, replace):
    files = snapshot(task)
    files["dispatch.md"] = files["dispatch.md"].replace(b"[root](root.yaml)", replace)
    with pytest.raises(schema.SchemaError):
        schema.check_design(files)


def test_skip_hardware_is_explicit_and_visible(task):
    task["root"]["CONTRACT"]["Hardware_specification"] = {"Device": "Unverified placeholder"}
    with pytest.raises(schema.SchemaError):
        schema.check_design(snapshot(task))
    result = schema.check_design(snapshot(task), skip_hardware=True)
    assert result.policy.skip_hardware
    assert any(item.category == "hardware.skipped" for item in result.diagnostics)


def test_no_external_locator_probe(task):
    with patch("subprocess.run", side_effect=AssertionError("checker ran a command")), patch(
            "importlib.util.find_spec", side_effect=AssertionError("checker discovered a parser")):
        schema.check_design(snapshot(task))
    assert not any(name == "kernelschema" or name.startswith("kernelschema.") for name in sys.modules)


def test_long_node_id_is_style_but_invalid_node_spelling_stays_hard(task):
    name = "branch_" + "x" * 51
    add_child(task, name)
    checked = schema.check_design(snapshot(task))
    assert any(item.category == "style.length" and name in item.path for item in checked.diagnostics)
    with pytest.raises(schema.SchemaError):
        schema.check_design(snapshot(task), style="strict")
    task[name]["META_DATA"]["Dispatch"]["id"] = "INVALID"
    with pytest.raises(schema.SchemaError):
        schema.check_design(snapshot(task))
