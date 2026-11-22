"""CPU writer integration through the real Agent/Claude native session protocol."""
from dataclasses import replace
import builtins
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aka.agent.provider import Agent, AgentOptions
from aka.agent.tools.client import call_tool
from aka.contracts.agent import AgentCapabilities
from aka.contracts.agent_resources import WorkspaceResourceContribution
from aka.contracts.workspace import RevisionContent
from aka.workflow.schema import materials
from aka.workflow.schema.language import collect_design
from aka.workflow.schema.planner import (Planner, PlannerConfig, PlannerRequest,
                                         _ReadTool, coder_context)

FIXTURE = Path(__file__).parent / "fixtures/schema_language/minimal"


class CpuAgent(Agent):
    """The actual native Agent, with an in-process fake process only.

    No shell/CLI runs in these tests. Production uses the real bwrap capability;
    the override allows us to inspect its requested mounts without needing it.
    """
    @property
    def capabilities(self):
        return AgentCapabilities(filesystem_isolation=True, tool_runtime=True)


def decision(status="replace"):
    return {"status": status, "rationale": "Use one scalar computation on the stated target",
            "hypothesis_key": "scalar-copy", "changed_execution_decisions": ["Copy the scalar directly"],
            "changes": [{"path": "root.DESIGN.Math_reshape", "before": "initial",
                         "after": "direct scalar copy", "reason": "no intermediate state"}],
            "evidence": [{"source": "task/public.md", "start": 1, "end": 1, "claim": "scalar contract"}],
            "next_actions": ["Implement the scalar operation and verify exact output"]}


def response(workspace, *, status="replace", files=None, text=False):
    choice = decision(status)
    if files is None:
        files = collect_design(FIXTURE) if status == "replace" else {}
    if text:
        return json.dumps({"decision": choice, "files": {name: raw.decode() for name, raw in files.items()}})
    (workspace / "output/decision.json").write_text(json.dumps(choice))
    for name, raw in files.items():
        path = workspace / "output/design" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return "completed"


@pytest.fixture
def harness(tmp_path):
    snapshot = materials.snapshot_inputs(tmp_path / "materials", materials.select_materials(wiki_ids=[]))
    actions, calls, admissions, usage = [], [], [], []
    def process(command, *, cwd, timeout, env):
        assert command[0] == "claude" and "--output-format" in command and "stream-json" in command
        assert env["ANTHROPIC_MODEL"] == "qwen3.8-max[1m]"
        assert timeout == 7
        workspace = Path(cwd)
        calls.append({"workspace": workspace, "prompt": command[-1], "environment": dict(env)})
        assert actions, "unexpected additional model call"
        action = actions.pop(0)
        final = action(workspace, env)
        return json.dumps({"type": "result", "result": final,
                           "usage": {"input_tokens": 11, "output_tokens": 7}}), "", 0, False
    agent = CpuAgent(AgentOptions(state_directory=tmp_path / "agent-state"), process_runner=process,
        environment={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "ANTHROPIC_MODEL": "qwen3.8-max[1m]"})
    planner = Planner(agent, snapshot, config=PlannerConfig(timeout_seconds=7, max_repairs=1),
                      observe_result=lambda op, result: usage.append((op, result.session_id, result.terminal_usage)))
    request = PlannerRequest("prepared", "design-1", 1, tmp_path / "operation", "Copy scalar x to y.\n", "Synthetic test device.\n")
    def admit(**kwargs):
        admissions.append(kwargs)
        return len(admissions)
    with agent.invocation_scope(admit=admit, observe=lambda *args, **kwargs: None):
        yield SimpleNamespace(planner=planner, request=request, actions=actions, calls=calls,
                              admissions=admissions, usage=usage, snapshot=snapshot, agent=agent)
    agent.close()


def first(h):
    h.actions.append(lambda workspace, env: response(workspace))
    return h.planner.design(h.request).design_ref


def optimization(h, ref, **kwargs):
    return replace(h.request, operation_id="optimize-1", directory=h.request.directory.parent / "optimization",
                   version=2, current_design=ref, **kwargs)


def test_initial_design_real_native_file_collection_and_no_wiki(harness):
    h = harness
    h.actions.append(lambda workspace, env: response(workspace))
    result = h.planner.design(h.request)
    assert result.status == "replace" and result.calls == 1
    assert materials.read_design(result.design_ref).checked.files == collect_design(FIXTURE)
    workspace = h.calls[0]["workspace"]
    assert not (workspace / "inputs/schema-materials/wiki").exists()
    selection = json.loads((workspace / "inputs/schema-materials/selection.json").read_text())
    assert selection["wiki"] == []
    access = h.admissions[0]["access"]
    assert workspace / "inputs" in access.read_only_paths
    assert all(not str(mount.source).endswith("KernelSchema") for mount in access.mounts)
    assert h.usage[0][0] == "design-1:attempt:1" and h.usage[0][2].total_tokens == 18
    assert (result.directory / "attempts/1/native/stdout.txt").read_text().find("completed") >= 0
    assert (result.directory / "attempts/1/raw/design/root.yaml").is_file()
    assert h.planner.design(h.request) == result
    assert len(h.calls) == 1


def test_claude_final_text_envelope_is_collected(harness):
    h = harness
    h.actions.append(lambda workspace, env: response(workspace, text=True))
    result = h.planner.design(h.request)
    assert result.status == "replace"
    assert materials.read_design(result.design_ref).checked.files == collect_design(FIXTURE)


@pytest.mark.parametrize("status", ["keep", "stop", "replace"])
def test_optimize_results_and_same_topology_math_rewrite(harness, status):
    h = harness
    ref = first(h)
    files = collect_design(FIXTURE)
    files["root.yaml"] = files["root.yaml"].replace(b"          y = x\n  Work_schedule:", b"          y = x + 0\n  Work_schedule:")
    h.actions.append(lambda workspace, env: response(workspace, status=status, files=files if status == "replace" else {}))
    result = h.planner.optimize(optimization(h, ref))
    assert result.status == status
    assert (result.design_ref is not None) == (status == "replace")
    if result.design_ref:
        assert result.design_ref != ref
        assert b"y = x + 0" in materials.read_design(result.design_ref).checked.files["root.yaml"]


def test_bad_draft_gets_bounded_new_metered_call_and_concrete_diagnostics(harness):
    h = harness
    h.actions.append(lambda workspace, env: response(workspace, files={"root.yaml": b"broken: true"}))
    def repair(workspace, env):
        text = (workspace / "inputs/task/repair-diagnostics.json").read_text()
        assert "dispatch.md" in text and "root.yaml" in text
        assert (workspace / "inputs/previous-output/design/root.yaml").read_bytes() == b"broken: true"
        return response(workspace)
    h.actions.append(repair)
    result = h.planner.design(h.request)
    assert result.status == "replace" and result.calls == 2
    assert len(h.admissions) == len(h.usage) == 2
    assert h.usage[0][1] != h.usage[1][1]
    assert (result.directory / "attempts/1/raw/design/root.yaml").read_bytes() == b"broken: true"


def test_completed_bad_draft_never_publishes(harness):
    h = harness
    h.actions.extend([lambda workspace, env: "completed", lambda workspace, env: "completed"])
    result = h.planner.design(h.request)
    assert result.status == "failed" and result.calls == 2 and result.design_ref is None
    assert not (result.directory / "published").exists()
    assert result.diagnostics


@pytest.mark.parametrize("extra", ["missing-provider", "extra-kernel", "empty-directory", "symlink"])
def test_whole_directory_closure_and_forbidden_entries(harness, extra):
    h = harness
    h.planner.config = replace(h.planner.config, max_repairs=0)
    def write(workspace, env):
        files = collect_design(FIXTURE)
        if extra == "missing-provider":
            files["dispatch.md"] += b"\n[branch](branch.yaml)\n"
        response(workspace, files=files)
        if extra == "extra-kernel":
            (workspace / "output/design/kernel.py").write_text("private code")
        elif extra == "empty-directory":
            (workspace / "output/design/extra").mkdir()
        elif extra == "symlink":
            (workspace / "output/design/leak.yaml").symlink_to(workspace / "inputs/task/public.md")
        return "completed"
    h.actions.append(write)
    result = h.planner.design(h.request)
    assert result.status == "failed" and result.design_ref is None


def test_equivalent_replace_is_keep_and_require_new_cannot_keep(harness):
    h = harness
    ref = first(h)
    h.actions.append(lambda workspace, env: response(workspace))
    result = h.planner.optimize(optimization(h, ref))
    assert result.status == "keep" and result.design_ref is None
    assert any(row["category"] == "design.equivalent" for row in result.diagnostics)
    request = replace(optimization(h, ref, require_new=True), directory=h.request.directory.parent / "new", operation_id="new")
    h.actions.extend([lambda workspace, env: response(workspace, status="keep"), lambda workspace, env: response(workspace)])
    result = h.planner.optimize(request)
    assert result.status == "failed" and result.calls == 2
    assert any("require_new" in row["message"] for row in result.diagnostics)


def test_allowed_normalization_is_saved_and_reread(harness):
    h = harness
    files = collect_design(FIXTURE)
    files["root.yaml"] = files["root.yaml"].replace(b"Math_reshape:", b"math_reshape:")
    h.actions.append(lambda workspace, env: response(workspace, files=files))
    result = h.planner.design(h.request)
    assert result.status == "replace" and result.calls == 1
    assert b"math_reshape:" in (result.directory / "attempts/1/raw/design/root.yaml").read_bytes()
    assert b"Math_reshape:" in materials.read_design(result.design_ref).checked.files["root.yaml"]
    corrections = json.loads((result.directory / "attempts/1/normalization.json").read_text())
    assert corrections["diffs"] and corrections["source_digest"] != corrections["normalized_digest"]


def test_frozen_public_view_cannot_expose_private_files(harness, tmp_path):
    h = harness
    secret = tmp_path / "private-reference.py"
    secret.write_text("DO_NOT_EXPOSE_PRIVATE_REFERENCE")
    snapshot = materials.snapshot_inputs(tmp_path / "with-unrelated",
        (*materials.select_materials(wiki_ids=[]), WorkspaceResourceContribution("private-mislabeled", secret, "private/reference.py")))
    h.planner.materials_snapshot = snapshot
    def write(workspace, env):
        assert not (workspace / "inputs/private").exists()
        assert "DO_NOT_EXPOSE" not in "".join(path.read_text() for path in (workspace / "inputs").rglob("*") if path.is_file())
        denied = json.loads(call_tool("schema.read_evidence", json.dumps({"source": str(secret)}).encode(), workspace=workspace, environment=env))
        assert denied["valid"] is False
        return response(workspace)
    h.actions.append(write)
    assert h.planner.design(h.request).status == "replace"


def test_schema_tools_use_real_transport_and_frozen_sources(harness):
    h = harness
    def write(workspace, env):
        response(workspace)
        checked = json.loads(call_tool("schema.check", b'{"source":"candidate"}', workspace=workspace, environment=env))
        assert checked["valid"] is True
        expanded = json.loads(call_tool("schema.expand", b'{"source":"candidate","node":"root"}', workspace=workspace, environment=env))
        assert "CONTRACT" in expanded["yaml"]
        read = json.loads(call_tool("schema.read_evidence", b'{"source":"task/public.md","start":1,"end":1}', workspace=workspace, environment=env))
        assert read["text"] == "Copy scalar x to y." and len(read["sha256"]) == 64
        return "completed"
    h.actions.append(write)
    assert h.planner.design(h.request).status == "replace"


def test_retained_raw_recovery_does_not_call_model_again(harness, monkeypatch):
    h = harness
    h.actions.append(lambda workspace, env: response(workspace))
    original = h.planner._collect
    with patch.object(h.planner, "_collect", side_effect=RuntimeError("host interrupted after raw receipt")):
        with pytest.raises(RuntimeError, match="host interrupted"):
            h.planner.design(h.request)
    assert h.planner.design(h.request).status == "replace"
    assert len(h.calls) == 1 and h.usage[0][1] == h.usage[1][1]
    assert h.usage[0][0] == h.usage[1][0]


def test_reused_provider_session_id_cannot_hide_a_new_repair_call(harness):
    h = harness
    h.actions.extend([lambda workspace, env: response(workspace, files={"root.yaml": b"bad: draft"}),
                      lambda workspace, env: response(workspace)])
    with patch("aka.agent.native.session_id_from_stream", return_value="provider-reused-session"):
        result = h.planner.design(h.request)
    assert result.status == "replace" and result.calls == 2
    assert h.usage[0][1] == h.usage[1][1]
    assert [row[0] for row in h.usage] == ["design-1:attempt:1", "design-1:attempt:2"]
    assert sum(row[2].total_tokens for row in h.usage) == 36


def test_native_public_observation_failure_keeps_original_result(harness):
    h = harness
    h.actions.append(lambda workspace, env: response(workspace))
    with patch("aka.agent.public_output.record_invocation", side_effect=OSError("public observation export interrupted")):
        with pytest.raises(OSError, match="public observation"):
            h.planner.design(h.request)
    assert (h.request.directory / "attempts/1/native/receipt.json").is_file()
    assert h.planner.design(h.request).status == "replace"
    assert len(h.calls) == 1 and len(h.usage) == 1


def test_started_unknown_call_is_never_relaunched(harness):
    h = harness
    h.actions.append(lambda workspace, env: (_ for _ in ()).throw(RuntimeError("lost process response")))
    with pytest.raises(RuntimeError, match="lost process"):
        h.planner.design(h.request)
    result = h.planner.design(h.request)
    assert result.status == "failed" and any(row["category"] == "invocation.unknown" for row in result.diagnostics)
    assert len(h.calls) == 1


@pytest.mark.parametrize("when", ["before", "after"])
def test_publication_crash_reuses_retained_candidate(harness, when):
    h = harness
    h.actions.append(lambda workspace, env: response(workspace))
    original = materials.durable_write_text
    def crash(path, *args, **kwargs):
        if path.name == "manifest.json":
            if when == "after":
                original(path, *args, **kwargs)
            raise OSError("publication crash")
        return original(path, *args, **kwargs)
    with patch.object(materials, "durable_write_text", side_effect=crash):
        with pytest.raises(RuntimeError, match="publication interrupted"):
            h.planner.design(h.request)
    result = h.planner.design(h.request)
    assert result.status == "replace" and len(h.calls) == 1
    assert result.design_ref.directory.name == ("published" if when == "after" else "published-recovery-1")


def test_operation_identity_and_private_role_isolation_guard(harness):
    h = harness
    first(h)
    with pytest.raises(ValueError, match="conflicts"):
        h.planner.design(replace(h.request, public_task="different contract"))
    with pytest.raises(ValueError, match="32 KiB"):
        h.planner.design(replace(h.request, feedback=({"text": "x" * 32768},)))


def test_partial_derived_write_recovers_without_model_retry(harness):
    from aka.workflow.schema import planner as module
    h = harness
    h.actions.append(lambda workspace, env: response(workspace))
    original = module.durable_write_text
    def crash(path, *args, **kwargs):
        if path.name == "root.yaml" and path.parent.name == "checked":
            raise OSError("derived-write crash")
        return original(path, *args, **kwargs)
    with patch.object(module, "durable_write_text", side_effect=crash):
        with pytest.raises(OSError, match="derived-write"):
            h.planner.design(h.request)
    assert h.planner.design(h.request).status == "replace"
    assert len(h.calls) == 1


@pytest.mark.parametrize("bad", [[], {}, None, 5])
def test_bad_decision_type_is_a_controlled_diagnostic(harness, bad):
    h = harness
    h.planner.config = replace(h.planner.config, max_repairs=0)
    def write(workspace, env):
        choice = decision()
        choice["status"] = bad
        (workspace / "output/decision.json").write_text(json.dumps(choice))
        return "completed"
    h.actions.append(write)
    result = h.planner.design(h.request)
    assert result.status == "failed" and result.diagnostics


def test_no_upstream_import_filesystem_or_subprocess_is_required(harness):
    h = harness
    native_import, native_open = builtins.__import__, os.open
    def guarded_import(name, *args, **kwargs):
        if name == "kernelschema" or name.startswith("kernelschema."):
            raise AssertionError("upstream import forbidden")
        return native_import(name, *args, **kwargs)
    def guarded_open(path, *args, **kwargs):
        if isinstance(path, (str, bytes, os.PathLike)) and "optional/KernelSchema" in str(path):
            raise AssertionError("upstream checkout read forbidden")
        return native_open(path, *args, **kwargs)
    h.actions.append(lambda workspace, env: response(workspace))
    with patch("builtins.__import__", side_effect=guarded_import), patch("os.open", side_effect=guarded_open), patch(
            "subprocess.run", side_effect=AssertionError("subprocess forbidden")), patch(
            "subprocess.Popen", side_effect=AssertionError("subprocess forbidden")):
        assert h.planner.design(h.request).status == "replace"
    assert not any(name == "kernelschema" or name.startswith("kernelschema.") for name in sys.modules)


def test_qualification_independent_fresh_session_and_all_obligations(harness):
    h = harness
    ref = first(h)
    def review(workspace, env):
        assert (workspace / "inputs/task/hardware.md").read_text() == h.request.target_hardware
        assert (workspace / "inputs/task/public.md").read_text() == h.request.public_task
        binding = json.loads((workspace / "inputs/review/binding.json").read_text())
        cite = {"source": "candidate/kernel.py", "start": 1, "end": 2, "claim": "direct scalar copy"}
        value = {"verdict": "pass", "design_identity": binding["design_identity"],
                 "source_identity": binding["source_identity"], "rationale": "all scalar obligations implemented",
                 "obligations": [{"id": name, "verdict": "pass", "reason": "copy without state",
                                  "evidence": [cite]} for name in binding["obligations"]],
                 "substantive_change": True, "change_evidence": [cite,
                     {"source": "initial-seed/kernel.py", "start": 1, "end": 1, "claim": "previous no-op"}]}
        (workspace / "output/decision.json").write_text(json.dumps(value))
        return "reviewed"
    h.actions.append(review)
    source = RevisionContent({"kernel.py": b"def copy(x):\n    return x\n"})
    result = h.planner.qualify(design_ref=ref, prepared_id="prepared", operation_id="review-1",
        directory=h.request.directory.parent / "review", source=source, source_identity="native-source-id",
        evidence={"seed_files": {"kernel.py": b"pass\n"}, "seed_identity": "native-seed-id",
                  "public_task": h.request.public_task, "target_hardware": h.request.target_hardware})
    assert result["qualified"] is True and result["substantive_change"] is True
    assert "not a formal proof" in result["basis"]
    assert len(result["evidence"]) == 6 and h.usage[0][1] != h.usage[1][1]
    assert "schema_implementation" in coder_context(ref, "prepared", 1, "seed-1")


def test_qualification_completion_or_self_approval_is_unknown(harness):
    h = harness
    ref = first(h)
    def review(workspace, env):
        (workspace / "output/decision.json").write_text(json.dumps({"qualified": True}))
        return "completed"
    h.actions.extend([review, review])
    result = h.planner.qualify(design_ref=ref, prepared_id="prepared", operation_id="review-1",
        directory=h.request.directory.parent / "review", source=RevisionContent({"kernel.py": b"pass\n"}),
        source_identity="native-source-id", evidence={"public_task": h.request.public_task,
                                                       "target_hardware": h.request.target_hardware})
    assert result["qualified"] is False and result["verdict"] == "unknown" and result["diagnostics"]
    assert result["calls"] == 2


def test_qualification_requires_independent_task_and_hardware(harness):
    h = harness
    ref = first(h)
    with pytest.raises(ValueError, match="native frozen public Task and target hardware"):
        h.planner.qualify(design_ref=ref, prepared_id="prepared", operation_id="review-1",
            directory=h.request.directory.parent / "review", source=RevisionContent({"kernel.py": b"pass\n"}),
            source_identity="native-source-id", evidence={})
    assert len(h.calls) == 1


def test_role_material_provenance_and_hardware_warning(tmp_path):
    snapshot = materials.snapshot_inputs(tmp_path / "snapshot", materials.select_materials("fp8_scale_group_gemm", max_wiki=1))
    view = materials.role_material_view(snapshot)
    selected = json.loads(view["schema-materials/selection.json"])
    assert selected["wiki"][0]["id"] == "fp8_scaled_grouped_gemm"
    assert selected["wiki"][0]["hardware_compatibility"] == "unproven"
    assert "Blackwell" in selected["guidance"] and "L20D" in selected["guidance"]
    assert selected["commit"] and selected["repository"]
    assert len(materials.select_materials("fp8 unrelated foo")) == 2


def test_role_view_rejects_material_name_spoofing(tmp_path):
    source = tmp_path / "secret"
    source.write_text("this is not an AKA specification")
    snapshot = materials.snapshot_inputs(tmp_path / "snapshot", (
        WorkspaceResourceContribution("spoof", source, "schema-materials/docs/operations.md"),))
    with pytest.raises(ValueError, match="unapproved or changed"):
        materials.role_material_view(snapshot)


def test_role_view_requires_complete_selected_wiki(tmp_path):
    selected = materials.select_materials(wiki_ids=["fp8_scaled_grouped_gemm"])
    snapshot = materials.snapshot_inputs(tmp_path / "snapshot", selected[:-1])
    with pytest.raises(ValueError, match="incomplete file closure"):
        materials.role_material_view(snapshot)
