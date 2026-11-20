import json
import os
from pathlib import Path
import subprocess
import sys


def test_default_native_path_has_no_optional_import_or_checkout_access(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    output = tmp_path / "off.json"
    subprocess.run([sys.executable, str(repo / "tests/schema_off_probe.py"), str(tmp_path / "run"), str(output)],
                   env={**os.environ, "PYTHONPATH": str(repo / "src")}, check=True, capture_output=True, text=True)
    trace = json.loads(output.read_text())
    assert trace["setup"] == trace["finish"] == 1
    assert trace["state"]["episodes"] == 2
    assert trace["state"]["tokens"] == 14
    assert len(trace["calls"]) == 2 and len(trace["evaluations"]) == 1
