"""Fresh-process negative guard around real host.run_selection with Schema off."""
import importlib.abc
import json
from pathlib import Path
import socket
import sys


class BlockSchema(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == "kernelschema" or fullname.startswith(("kernelschema.", "aka.workflow.schema")):
            raise AssertionError("off imported Schema: " + fullname)


def audit(event, args):
    if event in {"open", "os.listdir", "os.scandir"} and args:
        path = str(args[0]).replace("\\", "/")
        if "/workflow/schema/" in path or "/KernelSchema/" in path:
            raise AssertionError("off read Schema asset/parser: " + path)
    if event == "socket.connect" and args[0].family in {socket.AF_INET, socket.AF_INET6}:
        raise AssertionError("CPU probe attempted network")


sys.meta_path.insert(0, BlockSchema())
sys.addaudithook(audit)

from pytest import MonkeyPatch
from aka.bootstrap.host import run_selection
from aka.contracts.startup import Invocation
from aka.contracts.workflow_launch import encode_launch
from schema_public_fixture import install_native_cpu, launch, native_selection

with MonkeyPatch.context() as monkeypatch:
    harness = install_native_cpu(monkeypatch, Path(sys.argv[1]))
    if len(sys.argv) < 3 or sys.argv[2] == "host":
        result = run_selection(native_selection(harness), Invocation(encode_launch(launch(harness, max_iters=2))))
    else:
        from aka.bootstrap.workflow_cli import main
        monkeypatch.setenv("ANTHROPIC_MODEL", "qwen3.8-max[1m]")
        patch = harness.directory / "off.patch.json"
        patch.write_text(json.dumps({"api_version": 1, "patches": [
            {"id": "resources", "config": {"plugins": [], "bundles": []}},
            {"id": "workflow", "config": {"startup": True, "search": {}, "inherit_state": False,
                                          "output_limit_recovery_retries": 0,
                                          "episode_timeout_seconds": 60}}]}))
        (harness.directory / "plugins/gpu-wiki").mkdir(parents=True)
        (harness.directory / "gpu-wiki").mkdir()
        arguments = ["--repo-root", str(harness.directory), "--profile", "light", "--patch", str(patch),
            "--no-schema", "--workflow-request", encode_launch(launch(harness, max_iters=2))[1]]
        from importlib.metadata import entry_points
        if any(entry.name == "workflow" for entry in entry_points(group="aka.commands")):
            from aka.cli import main as public_main
            result = public_main(["workflow", *arguments])
        else:
            result = main(arguments)
    assert result == 0
    campaign, = harness.campaigns
    assert campaign.driver is None
    assert len(harness.calls) == 2
    assert all(row["kind"] == "implement" for row in harness.calls)
    assert not any(name.startswith("aka.workflow.schema") for name in sys.modules)
    print(json.dumps({"episodes": len(harness.calls), "schema_imported": False}))
