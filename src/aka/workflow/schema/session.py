"""Restricted native Agent profile and fixed-role invocation seam.

This is an AKA transport, not K3's qualified runtime. CPU tests exercise the
actual Agent/NativeSession implementation with a synthetic process runner.
The A5 owner retains native locators and receipts; no model is retried here.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from typing import Callable
from uuid import UUID

from aka.agent.backends import BackendAdapterRegistry
from aka.agent.backends.claude import ClaudeAdapter
from aka.agent.provider import Agent, AgentOptions
from kernelschema.contracts import ProtocolError, UnsupportedError, sha256
from kernelschema.runtime import ClaudeProfile, RoleOutput


NATIVE_PROFILE = "aka.schema.claude.no-tools.v1"


def native_capabilities():
    """Actual primitive surface; A5 must separately admit/qualify its business loop."""
    from kernelschema.contracts import PROTOCOL
    return {"protocols": [PROTOCOL], "operations": ["implement"], "modes": ["code", "verify_only"],
            "source_formats": ["single_file"], "empty_start": False,
            "recovery": ["new-session-from-explicit-source"], "token_enforcement": "unsupported",
            "extensions": []}


class RestrictedClaudeAdapter(ClaudeAdapter):
    """Reuse native session/usage adapters with K3's explicit no-tools flags."""
    supports_resume = False
    environment_variables = (*ClaudeAdapter.environment_variables, "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_SAFE_MODE", "DISABLE_AUTOUPDATER", "DISABLE_TELEMETRY", "AKA_SCHEMA_ROLE_SESSION")

    def __init__(self, profile):
        self.profile = profile
        self.authorities = {}
        self.inputs = {}
        self.deadlines = {}
        self.process_authorities = {}

    def build_command(self, prompt, session_id, reasoning_effort, settings):
        if settings:
            raise PermissionError("Schema native sessions cannot override CLI settings")
        if reasoning_effort != self.profile.effort:
            raise PermissionError("Schema native effort is fixed by its profile")
        if sha256(Path(self.profile.executable).read_bytes()) != self.profile.executable_sha256:
            raise ProtocolError("Schema native executable changed")
        binding = self.inputs.get(session_id)
        raw = prompt.encode("utf-8")
        if binding is None or sha256(raw) != binding[1] or len(raw) > self.profile.max_input_bytes:
            raise PermissionError("Schema prompt differs from its fixed native role input")
        # Native Agent owns the process. A tiny exec shim binds stdin without
        # placing complete Rules in Linux's size-limited single argv string.
        return [sys.executable, "-m", "aka.workflow.schema.stdin_exec", str(binding[0]),
                binding[1], self.profile.executable_sha256, "--", *self.profile.command(session_id)]

    def resume_command(self, command, session_id):
        raise UnsupportedError("this Schema native profile needs a fresh owner-authorized session")

    def prepare_environment(self, environment, *, reasoning_effort):
        key = environment.get("AKA_SCHEMA_ROLE_SESSION")
        bound = self.deadlines.get(key)
        if bound is not None:
            values = dict(environment)
            values["ATREX_AGENT_TIMEOUT_SECONDS"] = str(bound() / 1000)
            return values
        return environment

    @contextmanager
    def tool_environment(self, environment):
        # Original Agent invokes this hook immediately before its native launch.
        key = environment.get("AKA_SCHEMA_ROLE_SESSION")
        authorize = self.authorities.get(key)
        bound = self.deadlines.get(key)
        if bound is not None:
            bound()  # Original K3 absolute-cutoff check again at native launch.
        if authorize is None or not authorize():
            raise PermissionError("Schema owner denied this actual native call")
        authority = self.process_authorities.get(key)
        if authority is not None:
            environment = {**environment, "KS_BUDGET_AUTHORITY": str(authority.owner.path),
                "AKA_SCHEMA_OPERATION_INTENT": str(authority.context.root / ".runtime/intents" /
                    (authority.reservation["operation_id"] + ".json"))}
        yield environment


class SchemaNativeAgent(Agent):
    @property
    def configuration_identity(self):
        from kernelschema.contracts import digest
        return digest({"native": super().configuration_identity, "profile": NATIVE_PROFILE,
                       "claude": self.schema_adapter.profile.as_dict()})


def native_agent(profile: ClaudeProfile, *, process_runner=None, environment=None):
    """Create a separate original Agent owner with one restricted backend.

    No builtin tools, MCP, shell, hooks, project settings or native persistence.
    This is the trusted CLI's model permission boundary, not an OS sandbox for
    malicious executables. Actual profile qualification is required by A5/A7.
    """
    registry = BackendAdapterRegistry()
    adapter = RestrictedClaudeAdapter(profile)
    registry.register("claude", lambda _: adapter)
    original = os.environ if environment is None else environment
    allowed = {"HOME", "PATH", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR",
               "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ANTHROPIC_API_KEY",
               "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN"}
    if profile.auth_mode == "existing_cli_auth":
        allowed.add("CLAUDE_CONFIG_DIR")
    selected = {key: value for key, value in original.items() if key in allowed}
    selected.update(CLAUDE_CODE_SAFE_MODE="1", DISABLE_AUTOUPDATER="1", DISABLE_TELEMETRY="1")
    def unavailable(values):
        authority = adapter.process_authorities.get(values.get("AKA_SCHEMA_ROLE_SESSION"))
        if authority is not None and not authority.owner.check_active(authority.reservation):
            return 130, "original Schema owner paused, fenced or expired"
        return None
    agent = SchemaNativeAgent(AgentOptions(timeout_seconds=profile.timeout_ms / 1000), registry=registry,
                              environment=selected, process_runner=process_runner, unavailable=unavailable)
    agent.schema_adapter = adapter
    return agent


@dataclass(frozen=True)
class NativeRoleResult:
    """Preserves the complete native receipt, including unknown statistics."""
    invocation: object
    output: RoleOutput | None
    permission_errors: tuple[str, ...]
    assignment_sha256: str
    profile: str = NATIVE_PROFILE

    def native_source(self):
        """Exact original CandidateWorkspace.install_source input, only kernel.py."""
        from aka.contracts.workspace import RevisionContent
        if (self.permission_errors or self.output is None or self.output.role != "coder"
                or self.output.status != "ok" or self.output.assignment_sha256 != self.assignment_sha256):
            raise ProtocolError("no accepted coder transport output to install")
        return RevisionContent({"kernel.py": self.output.candidate_source})

    def freeze_source(self, publication, candidate_workspace, *, name="native-candidate"):
        """Freeze exact returned kernel bytes and return A2's actual SourceBinding.

        The caller owns draft installation/sealing and paid evaluation. This
        method neither changes a workspace nor turns a transport result into PASS.
        """
        source = self.native_source()
        from kernelschema.artifacts import Bundle
        from .source import bind_source
        reference = publication.put_source(name, source.files["kernel.py"])
        publication.commit([reference])
        return bind_source(Bundle(publication.root), reference, candidate_workspace)


class RoleSession:
    """One original native invocation using one fixed RoleContext.

    Owner callbacks authorize actual launches, retain the operation locator and
    absorb the full returned NativeInvocation. No callback creates a balance.
    A terminated retry/repair/reopen is a new owner operation with the same
    assignment and explicitly selected C. Unresolved original work is A5-fenced.
    """
    def __init__(self, agent, role, profile, *, workspace, session_id, output_directory,
                 authorize: Callable[[], bool], record_locator: Callable[[dict], None],
                 remaining_timeout_ms: Callable[[int], int] | None = None,
                 retain_invocation=None, maximum_timeout_ms=None):
        if not callable(authorize) or not callable(record_locator):
            raise TypeError("Schema native sessions require owner authorization and locator callbacks")
        try:
            if str(UUID(session_id)) != session_id:
                raise ValueError()
        except (ValueError, TypeError, AttributeError):
            raise ValueError("Schema native session_id must be its original canonical UUID") from None
        if not isinstance(agent, SchemaNativeAgent) or agent.schema_adapter.profile.as_dict() != profile.as_dict():
            raise PermissionError("Schema roles require the actual restricted native Agent profile")
        role.validate()
        selected = role.run.objects["profile"]
        if selected["parameters"].get("claude") != profile.as_dict() or selected["model"] != profile.requested_model:
            raise ProtocolError("native Agent profile/model differs from the fixed request")
        if role.document["role"] not in {"coder", "reviewer"} or role.run.request["operation"] != "implement":
            raise UnsupportedError("this AKA native profile admits implement coder/reviewer roles only")
        if role.document["source_sha256"] is None:
            raise UnsupportedError("this AKA native profile requires an explicitly selected current Source")
        if role.run.request.get("mode") == "verify_only" and role.document["role"] == "coder":
            raise PermissionError("verify_only cannot start a coder")
        limits = role.run.request["budget"]["limits"]
        if limits["calls"][role.document["role"]] == 0 or limits.get("model_calls") == 0 or limits["active_ms"] == 0:
            raise PermissionError("fixed Schema request prohibits this native role call")
        if limits["tokens"] is not None:
            raise UnsupportedError("Schema native no-tools profile has no hard token limit")
        maximum = min(profile.timeout_ms, limits["active_ms"])
        if maximum_timeout_ms is not None:
            if type(maximum_timeout_ms) is not int or maximum_timeout_ms <= 0:
                raise ProtocolError("native call timeout ceiling must be a positive integer")
            maximum = min(maximum, maximum_timeout_ms)
        self.role, self.profile, self.authorize = role, profile, authorize
        self.workspace = Path(workspace).absolute()
        if self.workspace.resolve(strict=True) != self.workspace:
            raise PermissionError("Schema role cwd cannot traverse symlinks")
        self.output_directory = Path(output_directory).resolve()
        self._check_scope()
        if self.output_directory == self.workspace or self.output_directory.is_relative_to(self.workspace):
            raise PermissionError("native observations must stay outside the role cwd")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        input_path = role.retain(self.output_directory)
        environment = {"AKA_SCHEMA_ROLE_SESSION": session_id,
                       "ATREX_AGENT_TIMEOUT_SECONDS": str(maximum / 1000)}
        if profile.auth_mode == "environment_auth":
            config = self.output_directory / "config"
            config.mkdir(exist_ok=True)
            environment["CLAUDE_CONFIG_DIR"] = str(config)
        if session_id in agent.schema_adapter.authorities:
            raise PermissionError("Schema native session authority already exists")
        agent.schema_adapter.authorities[session_id] = authorize
        agent.schema_adapter.inputs[session_id] = (input_path, sha256(role.payload))
        if remaining_timeout_ms is not None:
            agent.schema_adapter.deadlines[session_id] = lambda: remaining_timeout_ms(maximum)
        self._adapter = agent.schema_adapter
        self._session = agent.open_session("claude", self.workspace, session_id=session_id,
            environment=environment, output_directory=self.output_directory)
        if retain_invocation is not None:
            self._session.record_native_invocation = retain_invocation
        self._invoked = False
        # This callback must durably bind A5's existing operation/reservation.
        try:
            record_locator({"profile": NATIVE_PROFILE, "session_id": session_id,
                "workspace": str(self.workspace), "output_directory": str(self.output_directory),
                "assignment_sha256": role.assignment_sha256, "input_sha256": sha256(role.payload),
                "role": role.document["role"], "source_sha256": role.document["source_sha256"],
                "request_sha256": role.run.request_sha256,
                "maximum_timeout_ms": maximum,
                "timeout_ceiling_ms": maximum_timeout_ms,
                "native_context_resume": False, "process_recovery": "owner_required"})
        except BaseException:
            self.close()
            raise

    @classmethod
    def from_authority(cls, agent, role, profile, *, authority, **options):
        """A5 entry: callbacks and deadline come from one existing KS reservation."""
        from .authority import NativeCallAuthority
        if not isinstance(authority, NativeCallAuthority) or authority.context != role.run:
            raise ProtocolError("role and native authority must share the same original RunContext")
        session = cls(agent, role, profile, authorize=authority.authorize,
                      record_locator=authority.record_locator,
                      remaining_timeout_ms=authority.remaining_timeout_ms, **options)
        agent.schema_adapter.process_authorities[session.session_id] = authority
        return session

    def _check_scope(self):
        if self.workspace.is_symlink() or not self.workspace.is_dir() or any(self.workspace.iterdir()):
            raise PermissionError("Schema role cwd must be a fresh empty directory")
        for directory in self.workspace.parents:
            if any((directory / name).exists() for name in ("AGENTS.md", "CLAUDE.md", ".claude", ".mcp.json")):
                raise PermissionError("Schema role cwd has project customization ancestors")

    @property
    def session_id(self):
        return self._session.session_id

    def invoke(self, prompt="", *, purpose="initial", resume=False):
        if purpose not in {"initial", "repair", "rate_limit_retry", "resume", "reopen"}:
            raise ValueError("unknown Schema invocation purpose")
        if self._invoked or resume:
            raise UnsupportedError("reopen the original owner operation; this profile has no native resume/retry")
        self.role.validate()
        self._check_scope()
        # The old unrestricted workflow prompt is intentionally not delivered.
        # Feedback is an explicit RoleDelivery in the fixed RoleContext.
        self._invoked = True
        invocation = self._session.invoke(self.role.payload.decode("utf-8"),
                                          reasoning_effort=self.profile.effort)
        errors, output = self._inspect(invocation)
        return NativeRoleResult(invocation, output, tuple(errors), self.role.assignment_sha256)

    def _inspect(self, invocation):
        errors, initial, terminal = [], [], []
        if len(invocation.stdout.encode("utf-8")) > self.profile.max_output_bytes:
            return ["native_output_limit"], None
        for line in invocation.stdout.splitlines():
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError()
            except ValueError:
                errors.append("invalid_native_event")
                continue
            if event.get("type") == "system" and event.get("subtype") == "init":
                initial.append(event)
            if event.get("type") == "result":
                terminal.append(event)
            content = event.get("message", {}).get("content", []) if isinstance(event.get("message"), dict) else []
            if isinstance(content, list) and any(isinstance(item, dict) and item.get("type") in
                                                {"tool_use", "server_tool_use"} for item in content):
                errors.append("tool_use_forbidden")
        if len(initial) != 1:
            errors.append("missing_or_conflicting_permission_init")
        for event in initial:
            if event.get("tools") != [] or event.get("mcp_servers") != [] or event.get("skills"):
                errors.append("tool_permission_mismatch")
            plugins = event.get("plugins", [])
            if (not isinstance(plugins, list) or any(not isinstance(item, dict) or item.get("path") != "builtin"
                    or item.get("name") not in self.profile.builtin_plugins for item in plugins)
                    or {item.get("name") for item in plugins if isinstance(item, dict)} != set(self.profile.builtin_plugins)):
                errors.append("customization_permission_mismatch")
            if (event.get("permissionMode") != "dontAsk" or event.get("cwd") != str(self.workspace)
                    or event.get("session_id") != self.session_id or event.get("model") not in self.profile.reported_models):
                errors.append("native_profile_mismatch")
        output = None
        if len(terminal) != 1 or invocation.exit_status or invocation.timed_out or invocation.observation_errors:
            errors.append("native_terminal_unavailable")
        else:
            event = terminal[0]
            if (event.get("session_id") != self.session_id or event.get("subtype") != "success"
                    or event.get("is_error") is not False or event.get("permission_denials")):
                errors.append("native_terminal_failure")
            models = event.get("modelUsage")
            if models is not None and (not isinstance(models, dict)
                    or any(name not in self.profile.reported_models for name in models)):
                errors.append("native_reported_model_mismatch")
            try:
                output = RoleOutput.parse(event.get("structured_output", event.get("result")),
                    role=self.role.document["role"], assignment_sha256=self.role.assignment_sha256)
            except ProtocolError:
                errors.append("invalid_role_output")
        return sorted(set(errors)), None if errors else output

    def close(self):
        self._session.close()
        self._adapter.authorities.pop(self.session_id, None)
        self._adapter.inputs.pop(self.session_id, None)
        self._adapter.deadlines.pop(self.session_id, None)
        self._adapter.process_authorities.pop(self.session_id, None)
