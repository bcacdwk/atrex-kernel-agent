"""Durable Schema scheduling and admission, over the original native owners.

The root stores references and allowance, not a second Experiment/Job ledger.
An unknown invocation or dispatch prevents new work until its owner resolves it.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from threading import RLock

from aka.contracts.execution import JobResult
from aka.contracts.execution_recovery import InfrastructureRetryPolicy, JobReference
from aka.contracts.workspace import RevisionContent
from aka.core.filesystem import durable_write_json, ensure_private_directory, read_regular_bytes


class BudgetDenied(RuntimeError):
    """A configured allowance rejected work before launch."""


class RecoveryRequired(RuntimeError):
    """Retained owner facts need reconciliation; never an automatic replay."""


@dataclass(frozen=True)
class RunLimits:
    continuous: bool = False
    max_windows: int | None = None
    max_episodes: int | None = None
    max_agent_calls: int | None = None
    max_dispatches: int | None = None
    max_agent_active_seconds: float | None = None
    max_tokens: int | None = None
    per_call_active_seconds: float = 3600
    per_job_timeout_seconds: int = 3600
    window_agent_calls: int = 64
    window_dispatches: int = 128
    retry_delays: tuple[float, ...] = (1, 5, 15)
    planner_attempts: int = 3

    def __post_init__(self):
        if type(self.continuous) is not bool:
            raise TypeError("continuous must be explicit boolean")
        totals = (self.max_windows, self.max_episodes, self.max_agent_calls,
                  self.max_dispatches, self.max_tokens)
        for value in (*totals, self.per_job_timeout_seconds, self.window_agent_calls,
                      self.window_dispatches, self.planner_attempts):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError("count limits must be positive integers or unlimited")
        for value in (self.per_call_active_seconds, self.max_agent_active_seconds):
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise ValueError("active-time limits must be finite and positive")
        if self.per_call_active_seconds is None:
            raise ValueError("single invocation active-time limit is required")
        if not self.continuous and all(value is None for value in (*totals, self.max_agent_active_seconds)):
            raise ValueError("finite mode requires an explicit total allowance")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 or value > 60
               for value in self.retry_delays):
            raise ValueError("retry delays must be finite seconds between 0 and 60")

    def as_dict(self):
        return {**asdict(self), "retry_delays": list(self.retry_delays)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class RunState:
    def __init__(self, directory: Path, *, limits: RunLimits, workspace: Path, identity: dict):
        self.directory = Path(directory).absolute()
        self.path = self.directory / "run-state.json"
        self.limits = limits
        self.config = {"limits": limits.as_dict(), "workspace": str(Path(workspace).absolute()),
                       "identity": identity}
        self.root_ref = "schema:" + digest({"directory": str(self.directory), **self.config})
        self.lock = RLock()
        self._controls = {}
        self.data = None

    @contextmanager
    def owned(self):
        ensure_private_directory(self.directory)
        descriptor = os.open(self.directory / "writer.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RecoveryRequired("Schema root already has an active controller") from error
            try:
                self.data = json.loads(read_regular_bytes(self.path, limit=64 * 1024 * 1024))
            except FileNotFoundError:
                if any(item.name != "writer.lock" for item in self.directory.iterdir()):
                    raise RecoveryRequired("Schema root state is missing; retained work cannot receive fresh allowance")
                anchor = Path(self.config["workspace"]) / ".atrex_long_horizon/schema-controller.json"
                if anchor.exists():
                    raise RecoveryRequired("Native workspace retains a Schema root anchor; missing root state cannot reset allowance")
                self.data = {"schema": 1, "root_ref": self.root_ref, "config": self.config,
                    "phase": "prepare", "action": None, "action_sequence": 0, "prepared": None,
                    "design": None, "design_version": 0, "hypothesis": None, "hypotheses": {},
                    "champion": None, "tested_candidate": None, "working_draft": None,
                    "binding": None, "window": None, "windows": [], "episodes": {},
                    "calls": {}, "usage_receipts": {}, "diagnostics": [], "finalize": None,
                    "counts": {"windows": 0, "episodes": 0, "settled_episodes": 0,
                        "agent_calls": 0, "dispatches": 0, "agent_active_seconds": 0.0, "tokens": 0}}
                self.save()
            if (self.data.get("schema") != 1 or self.data.get("root_ref") != self.root_ref
                    or self.data.get("config") != self.config):
                raise RecoveryRequired("Schema root configuration/identity changed; explicit migration is required")
            anchor = Path(self.config["workspace"]) / ".atrex_long_horizon/schema-controller.json"
            expected = {"root_ref": self.root_ref, "directory": str(self.directory)}
            if anchor.exists():
                if json.loads(read_regular_bytes(anchor, limit=8192)) != expected:
                    raise RecoveryRequired("Native workspace belongs to a different Schema controller")
            else:
                durable_write_json(anchor, expected)
            yield self
        finally:
            os.close(descriptor)

    def save(self):
        with self.lock:
            durable_write_json(self.path, self.data)

    def diagnose(self, code, message):
        with self.lock:
            self.data["diagnostics"].append({"action": self.data["action"], "code": code, "message": str(message)})
            self.save()

    def action(self, kind):
        with self.lock:
            active = self.data["action"]
            if active is None:
                self.data["action_sequence"] += 1
                active = {"id": f"a{self.data['action_sequence']:08d}", "kind": kind}
                self.data["action"] = active
                self.save()
            elif active["kind"] != kind:
                raise RecoveryRequired("A different Schema action is still active")
            return active["id"]

    def pending(self):
        return {key: value for key, value in self.data["calls"].items() if value["state"] != "settled"}

    def require_allowance(self, counter, amount=1):
        limit = getattr(self.limits, "max_" + counter)
        if limit is not None and self.data["counts"][counter] + amount > limit:
            raise BudgetDenied("budget: " + counter)

    def require_new_work(self):
        if self.pending():
            raise RecoveryRequired("Unresolved native calls must be queried before any new work")
        self.require_allowance("tokens", 0)
        if self.limits.max_tokens is not None and self.data["counts"]["tokens"] >= self.limits.max_tokens:
            raise BudgetDenied("budget: observed tokens")

    def require_window_allowance(self, counter):
        window = self.data["window"]
        if window is not None and (self.data["counts"][counter] - window["start_counts"][counter]
                                  >= getattr(self.limits, "window_" + counter)):
            raise BudgetDenied("window budget: " + counter)

    def admit_agent(self, *, command, workspace, timeout, environment, backend, access, control):
        if timeout is None or timeout > self.limits.per_call_active_seconds:
            raise BudgetDenied("Agent invocation exceeds configured single-call active-time limit")
        # Enclosing ActiveBudget enforces elapsed active time in the original owner.
        with self.lock:
            reservations = sum(max(0, item["owner"]["active_limit"] - (item["active_seconds"] or 0))
                for item in self.pending().values() if item["kind"] == "agent")
            self.require_allowance("agent_active_seconds", float(timeout) + reservations)
            owner = {"workspace": str(workspace), "backend": backend,
                "command_digest": digest(list(command)), "active_limit": timeout,
                "home": environment.get("HOME")}
            if "--session-id" in command:
                index = list(command).index("--session-id") + 1
                if index < len(command):
                    owner["requested_session_id"] = command[index]
            permit = self._admit("agent", "agent_calls", owner)
            self._controls[permit] = control
            return permit

    def admit_execution(self, *, command, continuation=None):
        if command.timeout > self.limits.per_job_timeout_seconds:
            raise BudgetDenied("Execution exceeds configured per-job timeout")
        with self.lock:
            allowed = frozenset()
            if continuation is not None:
                from aka.execution.live_eval import LiveEval
                if not isinstance(continuation, LiveEval):
                    raise RecoveryRequired("Invalid live Eval authority")
                allowed = continuation.admit(self, command)
            permit = self._admit("execution", "dispatches", {"workspace": str(command.workspace),
                "operation": command.operation, "hardware": command.hardware,
                "command_digest": digest(list(command.command)), "timeout": command.timeout}, allowed=allowed)
            if continuation is not None:
                self.data["calls"][permit]["live_eval"] = dict(continuation.identity)
                self.data["calls"][permit]["original_permit"] = continuation.permits[0] if continuation.permits else permit
                self.save()
                continuation.permits.append(permit)
            return permit

    def _admit(self, kind, counter, owner, *, allowed=frozenset()):
        # A live model may synchronously call its tools. Only unresolved calls
        # inherited from a failed action bar admission; live parents are allowed.
        if any(item["state"] == "unknown" and permit not in allowed for permit, item in self.pending().items()):
            raise RecoveryRequired("Unknown owner state prevents another dispatch")
        self.require_allowance(counter)
        self.require_allowance("tokens", 0)
        if self.limits.max_tokens is not None and self.data["counts"]["tokens"] >= self.limits.max_tokens:
            raise BudgetDenied("budget: observed tokens")
        action = self.data["action"]
        if action is None:
            raise RecoveryRequired("An owned action is required before side effects")
        if action["kind"] == "implement":
            self.require_window_allowance(counter)
        ordinal = self.data["counts"][counter] + 1
        permit = f"{action['id']}:{kind}:{ordinal}"
        self.data["calls"][permit] = {"kind": kind, "state": "running", "action": action["id"],
            "owner": owner, "job_references": [], "active_seconds": None, "usage": "unavailable"}
        self.data["counts"][counter] = ordinal
        self.save()
        return permit

    def observe_agent(self, permit, *, event, value):
        with self.lock:
            call = self.data["calls"][permit]
            process = getattr(self._controls.get(permit), "scope", None)
            if process is not None:
                call["owner"]["process"] = {"pid": process.pid, "start_token": process.root_start}
            if event == "output":
                call["active_seconds"] = value.get("active_seconds")
            elif event == "result":
                call["state"] = "settled"
                call["active_seconds"] = value["active_seconds"]
                call["exit_status"], call["timed_out"] = value["result"][2:4]
            elif event == "error" and call["state"] != "settled":
                call["state"], call["error_type"] = "unknown", type(value).__name__
            elif event == "cancel_requested":
                call["cancel_requested"] = True
            if call["state"] == "settled" and getattr(self._controls.get(permit), "closed", True):
                self._controls.pop(permit, None)
            self.save()

    def observe_execution(self, permit, *, event, value, continuation=None):
        result = getattr(value, "execution_outcome", None) if event == "error" else value
        with self.lock:
            call = self.data["calls"][permit]
            if isinstance(result, JobResult):
                # Original owner persists full output. Root retains only the
                # confirmed recovery references and disposition needed to gate.
                call["job_references"] = [asdict(item) for item in result.job_references]
                call["disposition"] = result.facts.disposition
                call["transport_status"] = result.transport_status
                call["state"] = "settled" if (result.facts.disposition in {"terminal", "not_submitted"}
                    and all(item.disposition in {"terminal", "not_submitted"} for item in result.attempts)) else "unknown"
            else:
                call["state"] = "unknown"
            if event == "error":
                call["error_type"] = type(value).__name__
            if continuation is not None and continuation.association is not None:
                if (continuation.state is not self or not continuation.active
                        or continuation.scope.closed or permit not in continuation.permits):
                    raise RecoveryRequired("Live Eval association lost its original scope")
                for linked in continuation.permits:
                    original = self.data["calls"][linked]
                    if original.get("live_eval") != continuation.identity:
                        raise RecoveryRequired("Live Eval association changed owner")
                    original["live_eval_association"] = dict(continuation.association)
                    original["job_references"] = [continuation.association["job_reference"]]
                    if continuation.association["terminal"]:
                        original.update(state="settled", disposition="terminal")
            self.save()

    def reconcile(self, executor, *, recover_agent=None):
        """Only query original owners; query/cancel never buys a dispatch permit."""
        for permit, call in list(self.pending().items()):
            if call["kind"] == "execution" and call["job_references"]:
                results = [executor.query_job(JobReference(**ref), timeout=30,
                    retry=InfrastructureRetryPolicy((), 0)) for ref in call["job_references"]]
                if all(result.facts.disposition in {"terminal", "not_submitted"}
                       and all(item.disposition in {"terminal", "not_submitted"} for item in result.attempts)
                       for result in results):
                    with self.lock:
                        call["state"] = "settled"
                        call["disposition"] = "terminal"
                        self.save()
            elif call["kind"] == "agent" and recover_agent is not None:
                receipt = recover_agent(permit, dict(call))
                if isinstance(receipt, dict) and receipt.get("terminal") is True:
                    with self.lock:
                        elapsed = receipt.get("active_seconds")
                        elapsed = call["owner"]["active_limit"] if elapsed is None else elapsed
                        if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed < 0:
                            raise RecoveryRequired("Agent owner supplied invalid active usage")
                        call.update(state="settled", active_seconds=elapsed, recovery_basis=receipt.get("basis"))
                        # Charge only the unobserved remainder; never refund a
                        # previously checkpointed active-time observation.
                        old = call.get("charged_seconds", call.get("observed_active_seconds", 0))
                        self.data["counts"]["agent_active_seconds"] += max(0, elapsed - old)
                        call["charged_seconds"] = max(old, elapsed)
                        self.save()
        if self.pending():
            raise RecoveryRequired("Original owner still has an unresolved invocation/dispatch; inspect run-state.json")

    def checkpoint_active_time(self, used, inflight):
        with self.lock:
            # Attribute only this checkpoint's increase, separately for every
            # owned clock. Nested model/tool calls must not each inherit the
            # aggregate time of their parent or of earlier settled calls.
            delta = max(0.0, used - self.data["counts"]["agent_active_seconds"])
            observed = []
            for permit, control in self._controls.items():
                clock = getattr(control, "clock", None)
                if clock is not None:
                    call = self.data["calls"][permit]
                    observed.append((call, max(0.0, clock.active_seconds() - call.get("charged_seconds", 0))))
            total = sum(amount for _, amount in observed)
            # Sampling the clocks follows the aggregate sample by a few
            # microseconds; never attribute more than the saved root increase.
            fraction = min(1.0, delta / total) if total else 0.0
            for call, amount in observed:
                call["charged_seconds"] = call.get("charged_seconds", 0) + amount * fraction
            self.data["counts"]["agent_active_seconds"] = used
            self.data["active_time_inflight"] = inflight
            self.save()
            self._controls = {permit: control for permit, control in self._controls.items()
                              if not getattr(control, "closed", False)}

    def record_usage(self, receipt, tokens, measurement="unavailable"):
        with self.lock:
            value = {"tokens": tokens, "measurement": measurement}
            old = self.data["usage_receipts"].get(receipt)
            if old is not None:
                if old != value:
                    raise RecoveryRequired("Native usage receipt changed during replay")
                return
            if tokens is not None and (type(tokens) is not int or tokens < 0):
                raise ValueError("invalid native token usage")
            self.data["usage_receipts"][receipt] = value
            self.data["counts"]["tokens"] += tokens or 0
            self.save()

    def retain_source(self, source, *, owner, origin):
        identity = owner.source_identity(source)
        value = {"identity": identity, "files": {path: None if content is None else base64.b64encode(content).decode()
            for path, content in source.files.items()}}
        path = self.directory / "sources" / f"{identity}.json"
        if path.exists():
            if json.loads(read_regular_bytes(path, limit=128 * 1024 * 1024)) != value:
                raise RecoveryRequired("Retained source object changed")
        else:
            durable_write_json(path, value)
        return {"identity": identity, "object": str(path.relative_to(self.directory)), "origin": origin}

    def read_source(self, ref, *, owner):
        if ref["object"] != f"sources/{ref['identity']}.json":
            raise RecoveryRequired("Invalid source reference")
        value = json.loads(read_regular_bytes(self.directory / ref["object"], limit=128 * 1024 * 1024))
        source = RevisionContent({path: None if content is None else base64.b64decode(content, validate=True)
                                  for path, content in value["files"].items()})
        if value["identity"] != ref["identity"] or owner.source_identity(source) != ref["identity"]:
            raise RecoveryRequired("Retained source identity changed")
        return source
