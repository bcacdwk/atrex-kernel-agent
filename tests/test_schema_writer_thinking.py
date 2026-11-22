"""Writer-only settings through real Agent/NativeSession and root admission.

Only the process/model endpoint is synthetic. No external service or GPU runs.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aka.agent.native import NativeSession
from aka.agent.provider import AgentOptions
from aka.agent.tools.client import call_tool
from aka.contracts.workspace import RevisionContent
from aka.workflow.schema import materials
from aka.workflow.schema.launch import SchemaConfig
from aka.workflow.schema.planner import Planner, PlannerConfig, PlannerRequest
from aka.workflow.schema.run_state import RunLimits, RunState, digest
from test_schema_planner import CpuAgent, FIXTURE, response
from test_schema_qualification_protocol import review


@pytest.fixture
def writer(tmp_path, request, monkeypatch):
    options = dict(getattr(request, "param", {}))
    original_settings = options.pop("settings", {"permissions": {"allow": ["WebFetch"]},
        "skipWebFetchPreflight": True,
        "env": {"CLAUDE_CODE_MODEL_CAPABILITIES": "other-model=effort;qwen3.8-max=effort",
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000"}})
    env = {"HOME": str(tmp_path / "home"), "PATH": "/usr/bin:/bin",
           "ANTHROPIC_MODEL": "qwen3.8-max[1m]",
           options.pop("settings_key", "ATREX_CLAUDE_SESSION_SETTINGS"): json.dumps(original_settings)}
    env.update(options.pop("environment", {}))
    state = RunState(tmp_path / "root", limits=RunLimits(continuous=True),
        workspace=tmp_path / "native", identity={"synthetic-writer": True})
    snapshot = materials.snapshot_inputs(tmp_path / "materials", materials.select_materials(wiki_ids=[]))
    calls, actions, usage = [], [], []

    def process(command, *, cwd, timeout, env):
        workspace = Path(cwd)
        calls.append({"command": list(command), "environment": dict(env), "timeout": timeout,
                      "workspace": workspace})
        assert actions, "unexpected additional model invocation"
        final = actions.pop(0)(workspace, env)
        session_id = command[command.index("--session-id") + 1]
        # A dummy native ledger exercises NativeSession.close's real capture path.
        ledger = Path(env["HOME"]) / ".claude/projects/synthetic" / (session_id + ".jsonl")
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text('{"type":"synthetic-test-ledger"}\n')
        return json.dumps({"type": "result", "result": final,
            "usage": {"input_tokens": 11, "output_tokens": 7}}), "", 0, False

    sandbox = options.pop("sandbox", "none")
    if sandbox == "bwrap":
        # Keep real _prepare/prepare_home and scoped provider admission, replacing
        # only the final filesystem/process launch with the existing CPU seam.
        from aka.agent.execution.native import NativeExecution
        monkeypatch.setattr("aka.agent.execution.bubblewrap.BubblewrapExecution",
                            lambda *args, **kwargs: NativeExecution())
    agent = CpuAgent(AgentOptions(state_directory=tmp_path / "agent-state", sandbox=sandbox),
                     process_runner=process if sandbox == "none" else None, environment=env)
    if sandbox == "bwrap":
        monkeypatch.setattr(agent, "_process_runner", process)

    def observe(call_id, native):
        usage.append(call_id)
        state.record_usage("agent:" + call_id + ":" + native.session_id,
            native.terminal_usage.total_tokens, native.terminal_usage.measurement)

    planner = Planner(agent, snapshot, config=PlannerConfig(
        max_repairs=0, reasoning_effort="high", writer_thinking_budget_tokens=options.pop("budget", 8192)),
        observe_result=observe)
    assert not options
    request = PlannerRequest("prepared", "design-1", 1, tmp_path / "design-operation",
                             "Copy scalar x to y.\n", "Synthetic hardware.\n")
    with state.owned(), agent.invocation_scope(admit=state.admit_agent, observe=state.observe_agent):
        state.action("prepare")
        yield SimpleNamespace(agent=agent, planner=planner, request=request, state=state,
            calls=calls, actions=actions, usage=usage, settings=original_settings, environment=env)
    agent.close()


def settings(call):
    command = call["command"]
    return json.loads(command[command.index("--settings") + 1])


def initial(h):
    h.actions.append(lambda workspace, env: response(workspace))
    return h.planner.design(h.request)


@pytest.mark.parametrize("writer", [{}, {"settings_key": "ATREX_SESSION_SETTINGS"}], indirect=True)
def test_writer_override_preserves_settings_native_capture_and_root_charge(writer):
    h = writer
    identity = h.agent.configuration_identity
    result = initial(h)
    call = h.calls[0]
    expected = {**h.settings, "env": {**h.settings["env"], "MAX_THINKING_TOKENS": "8192",
        "CLAUDE_CODE_MODEL_CAPABILITIES": h.settings["env"]["CLAUDE_CODE_MODEL_CAPABILITIES"]
        + ";qwen3.8-max=-adaptive_thinking,-effort"}}
    assert settings(call) == expected
    assert h.agent.configuration_identity == identity
    assert call["timeout"] == 3600 and call["command"][call["command"].index("--effort") + 1] == "high"
    assert "MAX_THINKING_TOKENS" not in call["environment"]
    assert "CLAUDE_CODE_MODEL_CAPABILITIES" not in call["environment"]
    assert json.loads(call["environment"]["ATREX_CLAUDE_SESSION_SETTINGS"]) == expected
    intent = json.loads((result.directory / "intent.json").read_text())
    assert intent["config"]["writer_thinking_budget_tokens"] == 8192
    assert intent["writer_thinking"] == {"model": "qwen3.8-max[1m]",
        "mode": "fixed_budget_without_effort", "settings_sources": {}, "settings_sha256": hashlib.sha256(
        call["environment"]["ATREX_CLAUDE_SESSION_SETTINGS"].encode()).hexdigest()}
    assert json.loads((result.directory / "attempts/1/native/writer-thinking.json").read_text()) == intent["writer_thinking"]
    retained_call = next(iter(h.state.data["calls"].values()))
    assert retained_call["owner"]["command_digest"] == digest(call["command"])
    assert retained_call["state"] == "settled"
    assert h.state.data["counts"]["agent_calls"] == 1 and h.state.data["counts"]["tokens"] == 18
    manifests = list((result.directory / "attempts/1/native/.aka-agent-sessions").glob("*/session.json"))
    assert len(manifests) == 1 and json.loads(manifests[0].read_text())["runtime"] == "claude"
    assert h.planner.design(h.request) == result and len(h.calls) == 1


@pytest.mark.parametrize("writer", [{"budget": None}], indirect=True)
def test_default_command_settings_remain_identical(writer):
    result = initial(writer)
    assert settings(writer.calls[0]) == writer.settings
    assert "writer_thinking" not in json.loads((result.directory / "intent.json").read_text())
    assert "MAX_THINKING_TOKENS" not in settings(writer.calls[0])["env"]
    assert not (result.directory / "attempts/1/native/writer-thinking.json").exists()


@pytest.mark.parametrize("writer", [{"environment": {"ANTHROPIC_MODEL": "QWEN3.8-MAX[1M]"}}], indirect=True)
def test_capability_selector_normalizes_case_without_changing_model_selection(writer):
    result = initial(writer)
    assert writer.calls[0]["environment"]["ANTHROPIC_MODEL"] == "QWEN3.8-MAX[1M]"
    assert settings(writer.calls[0])["env"]["CLAUDE_CODE_MODEL_CAPABILITIES"].endswith(
        ";qwen3.8-max=-adaptive_thinking,-effort")
    assert json.loads((result.directory / "intent.json").read_text())["writer_thinking"]["model"] == "QWEN3.8-MAX[1M]"


def test_optimize_only_extract_review_and_coder_keep_original_settings(writer):
    h = writer
    first = initial(h)
    h.actions.append(lambda workspace, env: response(workspace, status="keep"))
    assert h.planner.optimize(replace(h.request, operation_id="optimize-1", version=2,
        current_design=first.design_ref, directory=h.request.directory.parent / "optimize")).status == "keep"
    assert settings(h.calls[-1])["env"]["MAX_THINKING_TOKENS"] == "8192"
    h.actions.append(lambda workspace, env: response(workspace, status="keep"))
    h.planner.retained_role(directory=h.request.directory.parent / "extract", inputs={"task": b"frozen"},
        prompt="Synthetic extraction", operation_id="extract-1", read_only_design=False)
    assert settings(h.calls[-1]) == h.settings
    h.actions.append(lambda workspace, env: review(workspace))
    result = h.planner.qualify(design_ref=first.design_ref, prepared_id="prepared", operation_id="qualify-1",
        directory=h.request.directory.parent / "qualify", source_identity="sealed-source",
        source=RevisionContent({"kernel.py": b"def copy(x):\n    return x\n"}),
        evidence={"public_task": h.request.public_task, "target_hardware": h.request.target_hardware,
                  "seed_files": {"kernel.py": b"pass\n"}, "seed_identity": "seed"})
    assert result["qualified"] and settings(h.calls[-1]) == h.settings
    h.actions.append(lambda workspace, env: "coded")
    workspace = h.request.directory.parent / "coder"
    workspace.mkdir()
    session = h.agent.open_session("claude", workspace,
        environment={"ATREX_AGENT_TIMEOUT_SECONDS": "3600"})
    assert isinstance(session, NativeSession)
    try:
        session.invoke("Synthetic Coder", reasoning_effort="high")
    finally:
        session.close()
    assert settings(h.calls[-1]) == h.settings
    assert h.state.data["counts"]["agent_calls"] == 5


@pytest.mark.parametrize("change", ["budget", "model", "settings", "effort"])
def test_changed_selection_rejects_cached_result_without_rewriting_intent(writer, change):
    h = writer
    result = initial(h)
    path = result.directory / "intent.json"
    before = path.read_bytes(), path.stat().st_mtime_ns
    if change == "budget":
        h.planner.config = replace(h.planner.config, writer_thinking_budget_tokens=4096)
    elif change == "effort":
        h.planner.config = replace(h.planner.config, reasoning_effort="medium")
    else:
        # A fresh native owner models a changed external selection, not mutation
        # of private Agent state or an existing runtime.
        env = dict(h.environment)
        env["ANTHROPIC_MODEL" if change == "model" else "ATREX_CLAUDE_SESSION_SETTINGS"] = (
            "different-concrete-model[1m]" if change == "model" else '{"skipWebFetchPreflight":false}')
        replacement = CpuAgent(h.agent.options, environment=env)
        if change == "model":
            # The existing global identity deliberately does not include model
            # environment values: the new writer intent must detect this itself.
            assert replacement.configuration_identity == h.agent.configuration_identity
        h.planner.agent = replacement
    try:
        with pytest.raises(ValueError, match="conflicts"):
            h.planner.design(h.request)
    finally:
        if change in {"model", "settings"}:
            h.planner.agent.close()
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before and len(h.calls) == 1


@pytest.mark.parametrize("budget", [True, False, 0, -1, 1, 1023, 16385, 32000, 8192.0, "8192", [], {}])
def test_invalid_budget_is_rejected_by_public_config(budget):
    with pytest.raises(ValueError, match="thinking budget"):
        SchemaConfig.load({"planner": {"writer_thinking_budget_tokens": budget}})


@pytest.mark.parametrize("backend", ["codex", "pi", "qoder", "dsh"])
def test_non_claude_budget_is_explicitly_unsupported(backend):
    with pytest.raises(ValueError, match="Claude backend"):
        PlannerConfig(agent_cli=backend, writer_thinking_budget_tokens=8192)


@pytest.mark.parametrize("writer", [
    {"environment": {"ANTHROPIC_MODEL": ""}},
    {"environment": {"ANTHROPIC_MODEL": "sonnet[1m]"}},
    {"environment": {"ANTHROPIC_MODEL": "claude-opus-4-20250514"}},
    {"environment": {"ANTHROPIC_MODEL": "model*"}},
    {"environment": {"ATREX_CLAUDE_SESSION_SETTINGS": "/tmp/not-inline-settings.json"}},
    {"environment": {"ATREX_CLAUDE_SESSION_SETTINGS": '{"env":{},"env":{}}'}},
    {"settings": {"model": "different-concrete-model"}},
    {"settings": {"env": []}},
    {"settings": {"env": {"CLAUDE_CODE_MODEL_CAPABILITIES": False}}},
    {"settings": []},
], indirect=True)
def test_ambiguous_or_invalid_model_settings_fail_before_intent_or_admission(writer):
    h = writer
    with pytest.raises(ValueError):
        h.planner.design(h.request)
    assert not h.request.directory.exists() and not h.calls
    assert h.state.data["counts"]["agent_calls"] == 0


def test_session_selection_change_after_intent_is_rejected_before_admission(writer, monkeypatch):
    h = writer
    original = h.agent.open_session
    def changed(*args, **kwargs):
        session = original(*args, **kwargs)
        session.environment["ANTHROPIC_MODEL"] = "changed-concrete-model"
        return session
    monkeypatch.setattr(h.agent, "open_session", changed)
    with pytest.raises(ValueError, match="changed after operation intent"):
        h.planner.design(h.request)
    assert not h.calls and h.state.data["counts"]["agent_calls"] == 0


def test_incomplete_starter_can_be_checked_then_completed_in_one_charged_call(writer):
    h = writer
    checks = []
    def draft(workspace, env):
        (workspace / "output/design/root.yaml").write_bytes(
            (workspace / "inputs/schema-materials/starter.yaml").read_bytes())
        (workspace / "output/design/dispatch.md").write_bytes((FIXTURE / "dispatch.md").read_bytes())
        checks.append(json.loads(call_tool("schema.check", b'{"source":"candidate"}',
                                           workspace=workspace, environment=env)))
        assert not checks[-1]["valid"] and checks[-1]["diagnostics"]
        assert not (h.request.directory / "published").exists()
        result = response(workspace)
        checks.append(json.loads(call_tool("schema.check", b'{"source":"candidate"}',
                                           workspace=workspace, environment=env)))
        assert checks[-1]["valid"]
        return result
    h.actions.append(draft)
    result = h.planner.design(h.request)
    assert result.status == "replace" and len(checks) == 2
    assert len(h.calls) == h.state.data["counts"]["agent_calls"] == 1


def test_unchanged_current_draft_does_not_satisfy_new_hypothesis(writer):
    h = writer
    first = initial(h)
    h.actions.append(lambda workspace, env: response(workspace))
    result = h.planner.optimize(replace(h.request, version=2, operation_id="optimize-new",
        directory=h.request.directory.parent / "new-hypothesis", current_design=first.design_ref, require_new=True))
    assert result.status == "failed" and result.design_ref is None
    assert not (result.directory / "published").exists()


def test_public_config_reaches_shared_native_owner_without_capping_coder(tmp_path, monkeypatch):
    from aka.agent.provider import Agent
    from aka.bootstrap.host import run_selection
    from aka.contracts.startup import Invocation
    from aka.contracts.workflow_launch import encode_launch
    from schema_public_fixture import install_native_cpu, launch, native_selection, schema_config
    observed = []
    original = Agent._execute
    def capture(self, command, cwd, timeout, env=None):
        observed.append(json.loads(command[command.index("--settings") + 1]))
        return original(self, command, cwd, timeout, env)
    monkeypatch.setattr(Agent, "_execute", capture)
    harness = install_native_cpu(monkeypatch, tmp_path)
    selected = native_selection(harness)
    baseline = {"skipWebFetchPreflight": True}
    selected = replace(selected, environment={**selected.environment,
        "ATREX_CLAUDE_SESSION_SETTINGS": json.dumps(baseline), "HOME": str(tmp_path / "agent-home")})
    configuration = schema_config(harness, episodes=1)
    configuration["planner"]["writer_thinking_budget_tokens"] = 8192
    assert run_selection(selected, Invocation(encode_launch(launch(harness))), schema=configuration) == 0
    campaign, = harness.campaigns
    assert campaign.driver.planner.agent is campaign.agent
    assert campaign.driver.planner.config.writer_thinking_budget_tokens == 8192
    assert [call["kind"] for call in harness.calls] == ["design", "implement"]
    assert observed == [{**baseline, "env": {"MAX_THINKING_TOKENS": "8192",
        "CLAUDE_CODE_MODEL_CAPABILITIES": "qwen3.8-max=-adaptive_thinking,-effort"}}, baseline]
    assert campaign.driver.state.data["counts"]["agent_calls"] == 2


def user_settings(h, value):
    path = Path(h.environment["HOME"]) / ".claude/settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.mark.parametrize("writer", [{"settings": {"skipWebFetchPreflight": True}}], indirect=True)
def test_user_settings_preserve_permissions_hooks_other_env_and_capabilities(writer):
    h = writer
    original = {"model": "qwen3.8-max[1m]", "permissions": {"deny": ["WebFetch"]},
        "hooks": {"SessionStart": []}, "env": {"UNRELATED_TEST_VALUE": "private-fixture-marker",
        "CLAUDE_CODE_MODEL_CAPABILITIES": "other-model=-adaptive_thinking;qwen3.8-max=effort"}}
    path = user_settings(h, original)
    raw = path.read_bytes()
    result = initial(h)
    intent = (result.directory / "intent.json").read_text()
    value = json.loads(intent)["writer_thinking"]
    assert value["settings_sources"] == {"user": hashlib.sha256(raw).hexdigest()}
    assert "private-fixture-marker" not in intent and '"hooks"' not in intent and '"permissions"' not in intent
    assert settings(h.calls[0]) == {"skipWebFetchPreflight": True, "env": {
        "MAX_THINKING_TOKENS": "8192",
        "CLAUDE_CODE_MODEL_CAPABILITIES": original["env"]["CLAUDE_CODE_MODEL_CAPABILITIES"]
        + ";qwen3.8-max=-adaptive_thinking,-effort"}}
    assert path.read_bytes() == raw
    assert h.planner.design(h.request) == result and len(h.calls) == 1


@pytest.mark.parametrize("value", [
    {"env": {"ANTHROPIC_MODEL": "claude-sonnet-4-6"}},
    {"model": "sonnet"},
    {"env": {"CLAUDE_CODE_MODEL_CAPABILITIES": "qwen3.8-max=-thinking"}},
    {"alwaysThinkingEnabled": False},
    {"env": {"MAX_THINKING_TOKENS": "4096"}},
    {"modelOverrides": {"qwen3.8-max": "some-other-model"}},
    {"availableModels": []},
    {"env": {"CLAUDE_CONFIG_DIR": "some-redirect"}},
    {"remoteSettingsHelper": "dynamic-command"},
])
def test_conflicting_or_dynamic_user_settings_reject_before_intent(writer, value):
    user_settings(writer, value)
    with pytest.raises(ValueError):
        writer.planner.design(writer.request)
    assert not writer.calls and not writer.request.directory.exists()
    assert writer.state.data["counts"]["agent_calls"] == 0


def test_user_settings_hash_change_rejects_replay_even_without_model_change(writer):
    path = user_settings(writer, {"skipWebFetchPreflight": True})
    result = initial(writer)
    intent = result.directory / "intent.json"
    original = intent.read_bytes(), intent.stat().st_mtime_ns
    path.write_text('{"skipWebFetchPreflight":false}')
    with pytest.raises(ValueError, match="conflicts"):
        writer.planner.design(writer.request)
    assert (intent.read_bytes(), intent.stat().st_mtime_ns) == original and len(writer.calls) == 1


def test_user_settings_changed_during_session_prepare_reject_before_admission(writer, monkeypatch):
    path = user_settings(writer, {"skipWebFetchPreflight": True})
    original = writer.agent.open_session
    def changed(*args, **kwargs):
        session = original(*args, **kwargs)
        path.write_text('{"skipWebFetchPreflight":false}')
        return session
    monkeypatch.setattr(writer.agent, "open_session", changed)
    with pytest.raises(ValueError, match="changed after operation intent"):
        writer.planner.design(writer.request)
    assert not writer.calls and writer.state.data["counts"]["agent_calls"] == 0


@pytest.mark.parametrize("writer", [{"budget": None}], indirect=True)
def test_default_does_not_inspect_or_reject_external_settings(writer):
    user_settings(writer, {"env": {"ANTHROPIC_MODEL": "different-model"}})
    assert initial(writer).status == "replace"
    assert settings(writer.calls[0]) == writer.settings


@pytest.mark.parametrize("kind", ["project", "local", "managed", "dropin", "remote", "legacy"])
def test_related_file_sources_are_hashed_and_conflicts_rejected(writer, monkeypatch, kind):
    from aka.workflow.schema import planner as module
    h = writer
    if kind in {"project", "local"}:
        path = h.request.directory.parent / ".claude" / ("settings.json" if kind == "project" else "settings.local.json")
    elif kind in {"managed", "dropin"}:
        root = h.request.directory.parent / "managed-settings"
        monkeypatch.setattr(module, "_CLAUDE_MANAGED_ROOT", root)
        path = root / ("managed-settings.json" if kind == "managed" else "managed-settings.d/10-policy.json")
    elif kind == "remote":
        path = Path(h.environment["HOME"]) / ".claude/remote-settings.json"
    else:
        path = Path(h.environment["HOME"]) / ".claude.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"permissions":{"deny":["WebFetch"]}}')
    result = initial(h)
    sources = json.loads((result.directory / "intent.json").read_text())["writer_thinking"]["settings_sources"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() in sources.values()
    path.write_text('{"env":{"ANTHROPIC_MODEL":"conflicting-model"}}')
    with pytest.raises(ValueError, match="conflicting"):
        h.planner.design(replace(h.request, directory=h.request.directory.parent / "new-operation", operation_id="new"))
    assert len(h.calls) == 1


@pytest.mark.parametrize("writer", [{"sandbox": "bwrap"}], indirect=True)
def test_native_prepared_home_rechecks_the_actual_copied_user_settings(writer):
    h = writer
    path = user_settings(h, {"env": {"ANTHROPIC_MODEL": "qwen3.8-max[1m]"},
                             "permissions": {"deny": ["WebFetch"]}})
    # prepare_home seeds this legacy file at HOME/.claude.json, but the CLI's
    # explicit private CLAUDE_CONFIG_DIR selects .claude/.claude.json instead.
    legacy = Path(h.environment["HOME"]) / ".claude.json"
    legacy.write_text('{"model":"legacy-not-read-by-the-isolated-cli"}')
    result = initial(h)
    env = h.calls[0]["environment"]
    copied = Path(env["CLAUDE_CONFIG_DIR"]) / "settings.json"
    assert Path(env["HOME"]) != Path(h.environment["HOME"])
    assert copied.read_bytes() == path.read_bytes()
    identity = json.loads((result.directory / "intent.json").read_text())["writer_thinking"]
    assert identity["settings_sources"] == {"user": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert settings(h.calls[0])["env"]["MAX_THINKING_TOKENS"] == "8192"
    assert h.planner.design(h.request) == result and len(h.calls) == 1


def test_oversize_settings_are_rejected_instead_of_hashing_a_valid_json_prefix(writer):
    path = user_settings(writer, {})
    path.write_bytes(b"{}" + b" " * (8 * 1024 * 1024))
    with pytest.raises(ValueError, match="bounded regular file"):
        writer.planner.design(writer.request)
    assert not writer.calls and not writer.request.directory.exists()


@pytest.mark.parametrize("writer", [{"sandbox": "bwrap"}], indirect=True)
def test_builtin_primary_agent_is_rejected_before_native_home_preparation(writer):
    user_settings(writer, {"agent": "statusline-setup", "permissions": {"allow": ["WebFetch"]}})
    with pytest.raises(ValueError, match="dynamic"):
        writer.planner.design(writer.request)
    assert not writer.request.directory.exists() and not writer.calls
    assert writer.state.data["counts"]["agent_calls"] == 0
    assert not (writer.request.directory.parent / "agent-state").exists()


def test_custom_primary_agent_is_rejected_without_parsing_frontmatter(writer):
    path = user_settings(writer, {"agent": "custom-primary"})
    definition = path.parent / "agents/custom-primary.md"
    definition.parent.mkdir()
    definition.write_text('---\nname: custom-primary\ndescription: Synthetic agent.\nmodel: sonnet\n---\nReply OK.\n')
    with pytest.raises(ValueError, match="dynamic"):
        writer.planner.design(writer.request)
    assert not writer.request.directory.exists() and not writer.calls
    assert writer.state.data["counts"]["agent_calls"] == 0


@pytest.mark.parametrize("writer", [{"settings": {"agent": "statusline-setup"}}], indirect=True)
def test_inline_primary_agent_is_rejected_before_intent_or_admission(writer):
    with pytest.raises(ValueError, match="dynamic"):
        writer.planner.design(writer.request)
    assert not writer.request.directory.exists() and not writer.calls
    assert writer.state.data["counts"]["agent_calls"] == 0


@pytest.mark.parametrize("writer", [{"budget": None}], indirect=True)
def test_default_does_not_reject_a_primary_agent_selection(writer):
    user_settings(writer, {"agent": "statusline-setup"})
    assert initial(writer).status == "replace"
    assert settings(writer.calls[0]) == writer.settings


@pytest.mark.parametrize("agent", ["", None])
def test_empty_primary_agent_selection_does_not_disable_budget(writer, agent):
    user_settings(writer, {"agent": agent, "skipWebFetchPreflight": True})
    assert initial(writer).status == "replace"
    assert settings(writer.calls[0])["env"]["MAX_THINKING_TOKENS"] == "8192"


def test_previous_budget_with_effort_intent_is_not_replayed_or_rewritten(writer, monkeypatch):
    from aka.workflow.schema import planner as module
    current = module._writer_thinking_settings
    def previous(*args, **kwargs):
        identity, override = current(*args, **kwargs)
        document = json.loads(override["ATREX_CLAUDE_SESSION_SETTINGS"])
        document["env"]["CLAUDE_CODE_MODEL_CAPABILITIES"] = document["env"][
            "CLAUDE_CODE_MODEL_CAPABILITIES"].removesuffix(",-effort")
        encoded = module._json(document).decode()
        identity.pop("mode")
        identity["settings_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
        return identity, {"ATREX_CLAUDE_SESSION_SETTINGS": encoded}
    with monkeypatch.context() as old:
        old.setattr(module, "_writer_thinking_settings", previous)
        result = initial(writer)
    intent = result.directory / "intent.json"
    before = intent.read_bytes(), intent.stat().st_mtime_ns
    assert "mode" not in json.loads(before[0])["writer_thinking"]
    with pytest.raises(ValueError, match="conflicts"):
        writer.planner.design(writer.request)
    assert (intent.read_bytes(), intent.stat().st_mtime_ns) == before
    assert len(writer.calls) == writer.state.data["counts"]["agent_calls"] == 1
    assert writer.state.data["counts"]["tokens"] == 18


@pytest.mark.parametrize("writer", [{"settings": {"skipWebFetchPreflight": True}}], indirect=True)
@pytest.mark.parametrize("kind", ["managed", "remote"])
@pytest.mark.parametrize("flags, supported", [
    ("adaptive_thinking", False),
    ("-adaptive_thinking,effort", False),
    ("-adaptive_thinking,-effort", True),
])
def test_managed_capabilities_must_already_allow_budget_without_effort(writer, monkeypatch, kind, flags, supported):
    from aka.workflow.schema import planner as module
    if kind == "managed":
        root = writer.request.directory.parent / "managed-settings"
        monkeypatch.setattr(module, "_CLAUDE_MANAGED_ROOT", root)
        path = root / "managed-settings.json"
    else:
        path = Path(writer.environment["HOME"]) / ".claude/remote-settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps({"env": {"CLAUDE_CODE_MODEL_CAPABILITIES": "qwen3.8-max=" + flags}}).encode()
    path.write_bytes(original)
    if supported:
        assert initial(writer).status == "replace"
        assert settings(writer.calls[0])["env"]["CLAUDE_CODE_MODEL_CAPABILITIES"].endswith(
            ";qwen3.8-max=-adaptive_thinking,-effort")
    else:
        with pytest.raises(ValueError, match="managed thinking or effort"):
            writer.planner.design(writer.request)
        assert not writer.request.directory.exists() and not writer.calls
        assert writer.state.data["counts"]["agent_calls"] == 0
    assert path.read_bytes() == original
