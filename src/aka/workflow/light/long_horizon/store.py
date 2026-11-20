from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aka.core.filesystem import durable_write_json

import aka.workflow.light.long_horizon.main_adapter as main_adapter
from aka.workflow.light.long_horizon.models import SupervisorState
from aka.workflow.light.long_horizon.protocol import atomic_write_json, atomic_write_text
from aka.workflow.light.long_horizon.telemetry import render_episode_brief

RUNTIME_DIR = ".atrex_long_horizon"
VERIFY_DIR = "verification_artifacts/.atrex_long_horizon_verify"
LIVE_MEMORY_FILE = "memory/live.json"


class CampaignStore:
    @staticmethod
    def checkpoint(workspace: Path) -> tuple[SupervisorState, dict[str, Any] | None]:
        """Read an explicit-episode checkpoint without creating a workspace.

        The ordinary supervisor retains its historical tolerant readers. A
        caller requesting an exact episode must not turn lost accounting into
        a fresh campaign, or recover a different episode accidentally.
        """
        root = Path(workspace).resolve() / RUNTIME_DIR
        def read(name):
            try:
                value = json.loads((root / name).read_text(encoding="utf-8"))
            except FileNotFoundError:
                return None
            if not isinstance(value, dict):
                raise ValueError(f"Invalid native checkpoint: {name}")
            return value

        raw = read("state.json")
        active = read("active_episode.json")
        if raw is None:
            if active is not None or any((root / "episodes").glob("*")):
                raise ValueError("Native accounting is missing; restore the original checkpoint")
            return SupervisorState(), None
        for name in ("episodes", "tokens"):
            if type(raw.get(name)) is not int or raw[name] < 0:
                raise ValueError(f"Invalid native checkpoint: {name}")
        attempts = raw.get("attempts")
        receipts = raw.get("usage_receipts")
        if not isinstance(attempts, list) or not isinstance(receipts, dict):
            raise ValueError("Invalid native attempt/usage checkpoint")
        numbers = []
        for attempt in attempts:
            if (not isinstance(attempt, dict) or type(attempt.get("episode")) is not int
                    or not 1 <= attempt["episode"] <= raw["episodes"]):
                raise ValueError("Invalid native attempt identity")
            numbers.append(attempt["episode"])
        if len(numbers) != raw["episodes"] or sorted(numbers) != list(range(1, len(numbers) + 1)):
            raise ValueError("Native terminal episode history is incomplete")
        if any(not isinstance(key, str) or type(value) is not int or value < 0
               for key, value in receipts.items()):
            raise ValueError("Invalid native usage receipts")
        if sum(receipts.values()) > raw["tokens"]:
            raise ValueError("Native token count is lower than its retained receipts")
        if active is not None:
            if (type(active.get("episode")) is not int
                    or active["episode"] not in {raw["episodes"], raw["episodes"] + 1}
                    or active["episode"] < 1
                    or type(active.get("memory_version")) is not int or active["memory_version"] < 1
                    or any(not isinstance(active.get(name), str) or not active[name]
                           for name in ("base_commit", "episode_branch", "worktree", "phase"))
                    or not Path(active["worktree"]).is_absolute()):
                raise ValueError("Invalid active native episode identity")
        return SupervisorState.from_dict(raw), active

    def __init__(self, workspace: Path, *, candidate_workspace):
        self.candidate_workspace = candidate_workspace
        self.workspace = workspace.resolve()
        self.root = self.workspace / RUNTIME_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.ensure_excluded(self.workspace, candidate_workspace=candidate_workspace)

    @staticmethod
    def ensure_excluded(workspace: Path, *, candidate_workspace, wiki_trace_only: bool = False) -> None:
        """Install supervisor excludes, or only the early Wiki trace subset.

        Workspace setup happens before the framework baseline commit.  At that
        point only consumer-owned Wiki telemetry may be excluded; installing
        the complete supervisor set would also hide baseline plans.
        """
        wiki_trace_rules = (
            "/.gpu_wiki_profile/",
            "/trace-retention-manifest.json",
            "/.aka-agent-observations/",
        )
        supervisor_rules = (
            f"/{RUNTIME_DIR}/",
            f"/{VERIFY_DIR}/",
            f"/{main_adapter.STALL_STATE_FILE}",
            f"/{LIVE_MEMORY_FILE}",
            *wiki_trace_rules,
            # Episode evidence is archived by the supervisor and must never
            # become part of the candidate commit.
            "/plans/",
            "/profiles/",
            "/.humanize/",
        )
        rules = wiki_trace_rules if wiki_trace_only else supervisor_rules
        from aka.contracts.workspace import WorkspaceExclusion
        candidate_workspace.exclude_runtime_paths(workspace, tuple(
            WorkspaceExclusion(rule.removeprefix('/').removesuffix('/'), subtree=rule.endswith('/'))
            for rule in rules))

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def active_path(self) -> Path:
        return self.root / "active_episode.json"

    @property
    def live_memory_path(self) -> Path:
        """Uncommitted, best-effort progress for the currently active episode."""
        return self.workspace / LIVE_MEMORY_FILE

    def episode_dir(self, episode: int) -> Path:
        path = self.root / "episodes" / f"e{episode:04d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load_state(self) -> SupervisorState:
        try:
            return SupervisorState.from_dict(json.loads(self.state_path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return SupervisorState()

    def save_state(self, state: SupervisorState) -> None:
        durable_write_json(self.state_path, state.as_dict())

    def record_usage(
        self, state: SupervisorState, receipt_id: str, cumulative_tokens: int
    ) -> None:
        """Persist usage and its replay receipt in one durable replacement."""
        previous = state.usage_receipts.get(receipt_id, 0)
        observed = max(previous, max(0, int(cumulative_tokens)))
        # Do not mutate the caller's state until the durable write succeeds.
        updated = SupervisorState.from_dict(state.as_dict())
        updated.tokens += observed - previous
        updated.usage_receipts[receipt_id] = observed
        self.save_state(updated)
        state.tokens = updated.tokens
        state.usage_receipts = updated.usage_receipts

    def load_active(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self.active_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def save_active(self, value: dict[str, Any]) -> None:
        atomic_write_json(self.active_path, value)

    def clear_active(self) -> None:
        self.active_path.unlink(missing_ok=True)

    def write_brief(self, episode: int, value: str) -> Path:
        path = self.episode_dir(episode) / "BRIEF.md"
        atomic_write_text(path, value)
        return path

    def archive_attempt(self, episode: int, value: dict[str, Any]) -> Path:
        path = self.episode_dir(episode) / "attempt.json"
        atomic_write_json(path, value)
        return path

    def archive_telemetry(self, episode: int, value: dict[str, Any]) -> Path:
        directory = self.episode_dir(episode)
        path = directory / "telemetry.summary.json"
        brief = render_episode_brief(value) + "\n"
        atomic_write_json(path, value)
        atomic_write_text(directory / "telemetry.brief.md", brief)
        return path
