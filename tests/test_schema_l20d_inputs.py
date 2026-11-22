"""Validate bundled public input identities/structure, never execute GPU sources."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

from aka.task.problem.inputs import validate_performance_baseline
from aka.task.problem.layout import validate_private_shapes
from aka.workflow.schema.language import CheckPolicy, check_design, loads

INPUTS = Path(__file__).resolve().parents[1] / "examples/schema/l20d"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name", ("vector_add", "fp8_scale_group_gemm"))
def test_public_input_hashes_stand_alone_without_upstream(name):
    base = INPUTS / name
    origin = json.loads((base / "ORIGIN.json").read_text())
    for relative, entry in origin["files"].items():
        assert digest(base / relative) == entry["sha256"]
    assert origin["new_gpu_verification"] is False
    if name == "vector_add":
        assert digest(base / "schema/root.yaml") == origin["adapted_schema_sha256"]
        assert digest(base / "schema/dispatch.md") == origin["adapted_dispatch_sha256"]
    else:
        assert digest(base / "benchmark/metadata.json") == origin["adapted_metadata_sha256"]


@pytest.mark.parametrize("name,count", (("vector_add", 3), ("fp8_scale_group_gemm", 10)))
def test_native_benchmark_structure_and_static_interfaces(name, count):
    base = INPUTS / name / "benchmark"
    shapes = validate_private_shapes(base / "shapes.json")
    metadata = json.loads((base / "metadata.json").read_text())
    assert len(shapes) == count
    assert all(str(int(key)) == key for key in shapes)
    assert len(validate_performance_baseline(shapes, metadata, "L20D")) == count
    module = ast.parse((base / "reference.py").read_text())
    model = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Model")
    assert any(isinstance(node, ast.FunctionDef) and node.name == "forward" for node in model.body)
    module = ast.parse((base / "input.py").read_text())
    assert any(isinstance(node, ast.FunctionDef) and node.name == "_make_inputs" for node in module.body)


def test_published_pair_is_exact_and_measurement_remains_historical():
    base = INPUTS / "vector_add"
    report = json.loads((base / "published/results.json").read_text())
    assert digest(base / "kernel.py") == report["source"]["candidate"]["sha256"]
    assert digest(base / "published/root.yaml") == report["source"]["schema"]["sha256"]
    assert report["task"]["platform"] == "L20D"
    assert report["host_abba"]["environment_identity"] is None
    assert report["host_abba"]["cross_record_comparability"] == "unknown"
    assert json.loads((base / "ORIGIN.json").read_text())["global_sota_claim"] is False


def test_schema_migration_retains_algorithm_tile_rounding_and_choices():
    base = INPUTS / "vector_add"
    original = loads((base / "published/root.yaml").read_text(), "root").value
    checked = check_design(base / "schema", policy=CheckPolicy(skip_hardware=True, style="advisory"))
    adapted = loads(checked.files["root.yaml"].decode(), "root").value
    for key in ("Input", "Output", "Compute"):
        assert adapted["CONTRACT"]["Algorithm_requirement"][key] == original["CONTRACT"]["Algorithm_requirement"][key]
    assert adapted["DESIGN"]["Math_reshape"] == original["DESIGN"]["Math_reshape"]
    old_step = original["DESIGN"]["Work_schedule"]["Step"]["MAIN"]
    step = adapted["DESIGN"]["Work_schedule"]["Step"]["MAIN"]
    assert step["Decomposition"] == old_step["Decomposition"]
    assert step["Mapping"]["Worker_count"] == old_step["Mapping"]["Worker_count"]
    assert step["Decomposition"]["Partition"]["elem"]["tile"] == 1024
    assert step["Schedule"]["Worker_assignment"]["Preassigned"] == {"start": "worker_id", "count": 1}
    for kind, entries in original["REALIZATION"].items():
        assert [entry["Choice"] for entry in adapted["REALIZATION"][kind]] == entries
    original_reference = original["CONTRACT"]["Algorithm_requirement"]["Golden_reference"]
    adapted_reference = adapted["CONTRACT"]["Algorithm_requirement"]["Golden_reference"]
    assert adapted_reference["Tolerance"]["y"] == original_reference["Tolerance"]["output"]
    assert "unknown" in adapted_reference["Tolerance"]["y"]
    old_role = original["DESIGN"]["Tile_pipeline"]["Step"]["MAIN"]["Role"]["MAIN"]
    role = adapted["DESIGN"]["Tile_pipeline"]["Step"]["MAIN"]["Role"]["MAIN"]
    assert role["Pipe"] == old_role["Pipe"]
    for verb, entries in old_role["Info"]["Using_verb"].items():
        for noun, entry in entries.items():
            assert role["Info"]["Using_verb"][verb][noun]["describe"] == entry["describe"]
    assert any(d.category == "hardware.skipped" for d in checked.diagnostics)


def test_grouped_gemm_adapter_preserves_source_and_all_historical_numbers():
    base = INPUTS / "fp8_scale_group_gemm"
    original = json.loads((base / "published/metadata.json").read_text())
    adapted = json.loads((base / "benchmark/metadata.json").read_text())
    assert all(adapted[key] == value for key, value in original.items())
    baseline = original["performance_baseline"]
    assert baseline["source_sha256"] == digest(base / "benchmark/reference.py")
    assert {key: value["production_performance"]["performance_us"] for key, value in adapted["shapes"].items()} == baseline["latency_us_by_shape"]
    assert adapted["aka_input_preparation"]["new_gpu_verification"] is False
    assert adapted["aka_input_preparation"]["fresh_v0_required"] is True
    assert not (base / "kernel.py").exists()
    assert not (base / "schema").exists()
