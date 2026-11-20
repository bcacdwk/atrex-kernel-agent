"""Finite native role calls, with durable receipts before returning to policy."""
from __future__ import annotations

from pathlib import Path
import json
from tempfile import mkdtemp
from types import SimpleNamespace
from uuid import uuid4

from kernelschema.contracts import ProtocolError
from kernelschema.runtime import ClaudeProfile

from .authority import NativeCallAuthority
from .context import RoleContext
from .session import RoleSession, NativeRoleResult, NATIVE_PROFILE


class Unsettled(ProtocolError):
    """Original work needs settlement, never another key or channel."""


class RoleCalls:
    def __init__(self, transactions, agent):
        self.tx, self.agent = transactions, agent
        self.run = transactions.context
        self.profile = ClaudeProfile(**self.run.objects["profile"]["parameters"]["claude"])

    def invoke(self, key, role, source, *, episode=None, deliveries=(), floor=None):
        op = self.tx.plan(key, role=role, source=source, episode=episode)
        output = self.run.root / ".runtime/aka-native" / op["id"]
        original = next((i for i in self.run.intents() if i["operation_id"] == op["id"]), None)
        if original:
            locator = original["native_locator"]
            context = RoleContext.reopen(self.run, output, expected_sha256=locator["input_sha256"])
            expected = RoleContext.prepare(self.run, role, source=source, deliveries=deliveries)
            if expected.payload != context.payload:
                raise ProtocolError("recovery cannot change original native role input")
            op["locator"] = locator
        else:
            context = RoleContext.prepare(self.run, role, source=source, deliveries=deliveries)
        self.tx.reserve(op, floor=floor)
        authority = NativeCallAuthority.bind(self.run, self.tx.owner, op["reservation"], source=source, episode=episode)
        invocation = self.tx.invocation(op)
        if invocation is None:
            owned = self.tx.owner.operations(self.run.run_id)[op["id"]]
            if owned["launched"]:
                self.tx.unknown(op, "original native call has no retained complete invocation")
                raise Unsettled(self.tx.state["stop_reason"])
            if "session_id" not in op:
                op.update(session_id=str(uuid4()), scope=mkdtemp(prefix="aka-schema-role-"))
                self.tx.save("native_scope_ready")
            try:
                session = RoleSession.from_authority(self.agent, context, self.profile,
                    authority=authority, workspace=Path(op["scope"]), session_id=op["session_id"],
                    output_directory=output, retain_invocation=lambda value: self.tx.retain_invocation(op, value))
            except BaseException:
                if not self.tx.owner.operations(self.run.run_id)[op["id"]]["launched"]:
                    self.tx.owner.release_unstarted(op["id"])
                raise
            try:
                op["locator"] = next(i for i in self.run.intents() if i["operation_id"] == op["id"])["native_locator"]
                self.tx.save("native_admitted")
                result = session.invoke(purpose="initial" if episode in (None, 1) else "repair")
                invocation = result.invocation
            except BaseException:
                owned = self.tx.owner.operations(self.run.run_id)[op["id"]]
                if owned["launched"] and self.tx.invocation(op) is None:
                    self.tx.unknown(op, "original native call interrupted before its durable receipt")
                elif not owned["launched"]:
                    self.tx.owner.release_unstarted(op["id"])
                raise
            finally:
                session.close()
        else:
            # Inspect original stream with the same A3 parser. This creates no
            # Agent session and asks no new model to reinterpret old history.
            inspector = object.__new__(RoleSession)
            inspector.role, inspector.profile = context, self.profile
            inspector.workspace = Path(op["locator"]["workspace"])
            inspector._session = SimpleNamespace(session_id=op["locator"]["session_id"])
            errors, answer = inspector._inspect(invocation)
            result = NativeRoleResult(invocation, answer, tuple(errors), context.assignment_sha256)
        reference = self.tx.absorb_invocation(op, invocation)
        models = set()
        for line in invocation.stdout.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                if event.get("type") == "system" and isinstance(event.get("model"), str):
                    models.add(event["model"])
                if isinstance(event.get("modelUsage"), dict):
                    models.update(event["modelUsage"])
        receipt = SimpleNamespace(ok=result.output is not None and not result.permission_errors,
            output=result.output, receipt=reference,
            facts={"runtime": NATIVE_PROFILE, "requested_model": self.profile.requested_model,
                   "reported_models": sorted(models),
                   "scope_id": op["session_id"]}, evidence_refs=lambda: [reference])
        return result, authority, op, receipt
