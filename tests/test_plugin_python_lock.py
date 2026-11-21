"""A workspace retains its plugin Python environment across CLI interpreters."""
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

import pytest


@pytest.fixture
def plugin_workspace(tmp_path):
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "schema.json").write_text(json.dumps({"type": "object"}))
    (plugin / "adapter.py").write_text(
        "import json, sys\n"
        "json.load(sys.stdin)\n"
        "print(json.dumps({'python': sys.executable, 'prefix': sys.prefix, 'argv': sys.argv[1:]}))\n"
    )
    manifest = {"id": "cpu-probe", "version": "1", "api_version": 1, "tools": {
        "query": {"description": "Report the actual CPU tool interpreter",
                  "command": ["{python}", "{plugin_root}/adapter.py", "{python}"],
                  "input_schema": "schema.json", "output_schema": "schema.json",
                  "timeout_seconds": 5}}}
    (plugin / "plugin.json").write_text(json.dumps(manifest))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    venv.EnvBuilder(with_pip=False, symlinks=True).create(tmp_path / "venv")
    other_python = str(tmp_path / "venv/bin/python")
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("ATREX_", "AKA_", "BASH_FUNC_"))}
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    def run(python, action, *, env=None):
        if action == "install":
            argv = ["-c", "from pathlib import Path; import sys; "
                    "from aka.resources.plugins import PluginRegistry; "
                    "PluginRegistry('', plugins=(sys.argv[1],), resources=(), agent_skills=())"
                    ".install(Path(sys.argv[2]))", str(plugin), str(workspace)]
        else:
            argv = ["-m", "aka.resources.cli", "--plugin-dir", "", "--plugin", str(plugin),
                    "--workspace", str(workspace), *action]
        return subprocess.run([python, *argv], input="{}", text=True, capture_output=True,
                              cwd=workspace, env={**environment, **(env or {})})

    return plugin, workspace / ".atrex_plugins/lock.json", manifest, other_python, run


@pytest.mark.parametrize("venv_writer", [True, False])
def test_workspace_load_and_call_keep_locked_python(plugin_workspace, venv_writer):
    _plugin, lock, _manifest, other_python, run = plugin_workspace
    writer, reader = (other_python, sys.executable) if venv_writer else (sys.executable, other_python)
    assert writer != reader
    installed = run(writer, "install")
    assert installed.returncode == 0, installed.stderr
    original = lock.read_bytes()
    assert json.loads(original)["plugins"][0]["commands"]["query"][0] == writer

    for python in (reader, writer):
        listed = run(python, ["list"])
        assert listed.returncode == 0, listed.stdout + listed.stderr
        assert "cpu-probe.query" in listed.stdout
        called = run(python, ["call", "cpu-probe.query", "--input", "-"])
        assert called.returncode == 0, called.stdout + called.stderr
        result = json.loads(called.stdout)
        assert result["python"] == writer
        assert result["argv"] == [writer]
        if venv_writer:
            assert result["prefix"] == str(Path(other_python).parent.parent)
        assert lock.read_bytes() == original


@pytest.mark.parametrize("change", ["adapter", "schema", "version", "command", "literal_python"])
def test_cross_python_load_rejects_real_plugin_changes(plugin_workspace, change):
    plugin, lock, manifest, other_python, run = plugin_workspace
    installed = run(other_python, "install")
    assert installed.returncode == 0, installed.stderr
    original = lock.read_bytes()
    if change == "adapter":
        with (plugin / "adapter.py").open("a") as stream:
            stream.write("# changed plugin content\n")
    elif change == "schema":
        (plugin / "schema.json").write_text(json.dumps({"type": "object",
            "properties": {"value": {"type": "string"}}, "required": ["value"]}))
    else:
        if change == "version":
            manifest["version"] = "2"
        elif change == "command":
            manifest["tools"]["query"]["command"].append("changed")
        else:
            manifest["tools"]["query"]["command"][0] = sys.executable
        (plugin / "plugin.json").write_text(json.dumps(manifest))
    result = run(sys.executable, ["list"])
    assert result.returncode == 1
    assert json.loads(result.stdout)["error"]["code"] == "plugin_changed"
    assert lock.read_bytes() == original


def test_missing_locked_python_does_not_use_current_python(plugin_workspace):
    _plugin, lock, _manifest, other_python, run = plugin_workspace
    installed = run(other_python, "install")
    assert installed.returncode == 0, installed.stderr
    original = lock.read_bytes()
    Path(other_python).unlink()
    result = run(sys.executable, ["list"])
    assert result.returncode == 1
    assert json.loads(result.stdout)["error"]["code"] == "plugin_changed"
    assert lock.read_bytes() == original


def test_path_command_rebinding_is_still_rejected(plugin_workspace, tmp_path):
    plugin, lock, manifest, other_python, run = plugin_workspace
    manifest["tools"]["query"]["command"] = ["selected-python", "{plugin_root}/adapter.py"]
    (plugin / "plugin.json").write_text(json.dumps(manifest))
    paths = [tmp_path / "first-bin", tmp_path / "second-bin"]
    for path in paths:
        path.mkdir()
        (path / "selected-python").symlink_to(sys.executable)
    installed = run(other_python, "install", env={"PATH": str(paths[0])})
    assert installed.returncode == 0, installed.stderr
    original = lock.read_bytes()
    result = run(sys.executable, ["list"], env={"PATH": str(paths[1])})
    assert result.returncode == 1
    assert json.loads(result.stdout)["error"]["code"] == "plugin_changed"
    assert lock.read_bytes() == original


def test_python_binding_still_compares_remaining_locked_argv(plugin_workspace):
    _plugin, lock, _manifest, other_python, run = plugin_workspace
    installed = run(other_python, "install")
    assert installed.returncode == 0, installed.stderr
    altered = json.loads(lock.read_text())
    altered["plugins"][0]["commands"]["query"].append("unexpected-argument")
    lock.write_text(json.dumps(altered))  # Only this test's new fixture lock.
    original = lock.read_bytes()
    result = run(sys.executable, ["list"])
    assert result.returncode == 1
    assert json.loads(result.stdout)["error"]["code"] == "plugin_changed"
    assert lock.read_bytes() == original
