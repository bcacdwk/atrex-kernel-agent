"""Run selected CPU regressions from a copied fixture bundle against wheels.

Invoke with an isolated interpreter (`python -I`), no PYTHONPATH, and arguments:
  <fixture-directory> <work-directory> <forbidden-source-root> [...]
Only test/fixture files are copied; aka must come from this interpreter's prefix.
"""
import importlib.abc
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys

fixtures, work = (Path(value).resolve() for value in sys.argv[1:3])
forbidden = tuple(Path(value).resolve() for value in sys.argv[3:])
assert not os.environ.get("PYTHONPATH"), "installation check must not inherit PYTHONPATH"
assert sys.flags.isolated, "run the installed probe with python -I"
assert all(not any(Path(value or ".").resolve().is_relative_to(root) for root in forbidden)
           for value in sys.path)


class BlockUpstream(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == "kernelschema" or fullname.startswith("kernelschema."):
            raise AssertionError("installed code attempted KernelSchema import: " + fullname)


def audit(event, args):
    if event in {"open", "os.listdir", "os.scandir", "os.chdir"} and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).absolute()
        if any(path.is_relative_to(root) for root in forbidden):
            raise AssertionError("installed code accessed forbidden source tree: " + str(path))
        if "KernelSchema" in path.parts:
            raise AssertionError("installed code accessed upstream checkout: " + str(path))
    if event == "socket.connect" and args[0].family in {socket.AF_INET, socket.AF_INET6}:
        raise AssertionError("installed CPU check attempted network")
    if event == "subprocess.Popen":
        executable, command = os.fsdecode(args[0]), args[1]
        if Path(executable).name == "git":
            assert not set(command) & {"clone", "fetch", "pull", "push", "ls-remote"}, "network Git is forbidden"
        elif Path(executable).resolve() != Path(sys.executable).resolve():
            raise AssertionError("CPU check launched unexpected external program: " + executable)


sys.meta_path.insert(0, BlockUpstream())
sys.addaudithook(audit)
sys.path.insert(0, str(fixtures))
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
os.environ["PIP_NO_INDEX"] = "1"

import aka
import aka.workflow.schema as schema
from aka.workflow.schema import materials, release
from aka.workflow.schema.language import CheckPolicy, check_design
from aka.workflow.schema.planner import Planner
from aka.workflow.schema.workflow import extract, implement

prefix = Path(sys.prefix).resolve()
assert Path(aka.__file__).resolve().is_relative_to(prefix)
assert Path(schema.__file__).resolve().is_relative_to(prefix)
assert materials.ASSETS.resolve().is_relative_to(prefix)
assert callable(Planner.design) and callable(Planner.optimize)
assert callable(extract) and callable(implement)
assert not any(dist.metadata["Name"].lower().replace("_", "-") == "kernelschema"
               for dist in importlib.metadata.distributions())
for name in ("atrex-aka-framework", "atrex-aka-legacy", "atrex-aka-light", "atrex-aka-schema"):
    distribution = importlib.metadata.distribution(name)
    direct = distribution.read_text("direct_url.json")
    if direct is not None:
        origin = json.loads(direct)
        assert "dir_info" not in origin, "editable/source-directory installation is forbidden"
        assert origin.get("url", "").endswith(".whl"), "install from a built wheel"
        assert not any(str(root) in origin["url"] for root in forbidden)
    assert not any("kernelschema" in item.lower() for item in distribution.requires or ())
for pth in prefix.rglob("*.pth"):
    text = pth.read_text()
    assert not any(str(root) in text for root in forbidden), str(pth)

identity = release.verify_release()
entries = json.loads((materials.ASSETS / "wiki-index.json").read_text())["entries"]
nodes = 0
for entry in entries:
    directory = materials.ASSETS / "ks/schema_wiki" / entry["directory"]
    # Historical extraction receipts are provenance beside the actual flat
    # design. Project every declared node, then exercise the directory reader.
    projected = work.parent / "wiki" / entry["id"]
    projected.mkdir(parents=True, exist_ok=True)
    for name in entry["files"]:
        (projected / name).write_bytes((directory / name).read_bytes())
    checked = check_design(projected, policy=CheckPolicy(skip_hardware=True))
    assert set(checked.files) == set(entry["files"])
    nodes += len(checked.resolved.raw_task)
assert len(entries) == 18
assert (Path(schema.__file__).parent / "NOTICE.md").is_file()
assert (materials.ASSETS / "SOURCES.json").is_file()
assert (materials.ASSETS / "DOCUMENT_VIEWS.json").is_file()

print(json.dumps({"aka_file": aka.__file__, "schema_file": schema.__file__,
    "assets": str(materials.ASSETS), "wiki_directories": len(entries), "wiki_nodes": nodes,
    "release": identity["manifest_sha256"], "four_apis_available": True}, sort_keys=True), flush=True)

import pytest
selected = [
    "test_schema_public_entrypoints.py",
    "test_schema_planner.py::test_initial_design_real_native_file_collection_and_no_wiki",
    "test_schema_planner.py::test_optimize_results_and_same_topology_math_rewrite",
    "test_schema_extract_stages.py::test_single_root_all_stages_fresh_native_sessions_raw_and_projection_replay",
    "test_schema_extract_stages.py::test_multi_node_p3_accepts_unwritten_plan_and_p5_registered_only_after_review",
]
raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", "--disable-warnings", "--maxfail=2",
    "--basetemp", str(work), *(str(fixtures / name) for name in selected)]))
