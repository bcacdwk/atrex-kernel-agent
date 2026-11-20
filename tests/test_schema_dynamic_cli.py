"""Installed CLI selection; writing subprocess is an explicit synthetic script."""
from pathlib import Path
import subprocess
import sys

import pytest

from kernelschema.contracts import loads
from test_schema_dynamic import dynamic, fixed, close


@pytest.mark.parametrize("command,start", [("workflow", "schema"), ("optimize", "design")])
def test_actual_cli_dynamic_selection_and_original_parent(dynamic, tmp_path, command, start):
    d = dynamic(start=start, optimize=False)
    close(d)
    repo = Path(__file__).resolve().parents[1]
    args = [sys.executable, "-I", "-m", "aka", command, "--repo-root", str(repo),
            "--problem", str(Path(d.selection.document["request"]).parent),
            "--workspace-base", str(tmp_path / "native-cli"), "--platform", "H20",
            "--framework", "triton", "--framework-baseline", "never", "--convert-after", "0",
            "--token-budget", "0", "--schema", "--schema-config", str(d.selection.origin)]
    child = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True, timeout=90)
    assert child.returncode == 0, child.stdout + child.stderr
    result = loads((d.workflow.root / "dynamic-result.json").read_bytes())
    assert result["format"] == "aka.dynamic.v1" and result["status"] == "blocked"
    assert result["optimize_count"] == 0 and result["owner"] == d.owner.descriptor
    assert len(result["history"]) == (2 if start == "design" else 1)
    # Native transport is deliberately unqualified; no real model/GPU call.
    assert all(op["role"] in {"planner", "reviewer"} for op in d.owner.operations().values())
    before = d.owner.operations()
    child = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True, timeout=90)
    assert child.returncode == 0, child.stdout + child.stderr
    assert d.owner.operations() == before


def test_dynamic_selection_preserves_factory_preimport_hash_boundary(dynamic, tmp_path):
    d = dynamic(optimize=False)
    close(d)
    script = """
import sys
from aka.workflow.schema.provider import SchemaSelection
selection = SchemaSelection.load(sys.argv[1])
assert 'aka.workflow.schema.backend' not in sys.modules
selection.backend_factory()
assert 'aka.workflow.schema.backend' in sys.modules
"""
    child = subprocess.run([sys.executable, "-I", "-c", script, str(d.selection.origin)],
                           cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stdout + child.stderr
