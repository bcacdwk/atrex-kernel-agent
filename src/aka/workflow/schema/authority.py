"""Bind native callbacks to one existing KS reservation; no account/ledger here."""
from __future__ import annotations

from dataclasses import dataclass

from kernelschema.budget import OwnerAccount
from kernelschema.contracts import ProtocolError, jcs, loads


@dataclass(frozen=True)
class NativeCallAuthority:
    context: object
    owner: OwnerAccount
    facts: bytes

    @classmethod
    def bind(cls, context, owner, reservation, *, source, episode=None):
        """Attach a prior reserve/reserve_group member; never allocate one.

        A5 owns the original account, reservation, role, finish floor and group
        claim. The returned methods fit RoleSession's existing callback slots.
        """
        if not isinstance(owner, OwnerAccount) or not isinstance(reservation, dict):
            raise TypeError("native authority needs an existing KS owner and reservation")
        bound = OwnerAccount.for_context(context, create=False, authority=owner.path)
        if bound.path != owner.path or bound.descriptor != owner.descriptor:
            raise ProtocolError("native owner differs from the original context authority")
        if reservation.get("run_id") != context.run_id or reservation.get("owner") != context.owner["id"]:
            raise ProtocolError("native reservation belongs to another original run/owner")
        operation = owner.operations(context.run_id).get(reservation.get("operation_id"))
        if operation is None or operation["reservation"] != reservation:
            raise ProtocolError("native reservation is not the original owner's durable operation")
        if operation["role"] not in {"coder", "reviewer"}:
            raise ProtocolError("native role operation must be coder or reviewer")
        context.bundle.source(source)
        return cls(context, owner, jcs({"reservation": reservation, "source": source, "episode": episode,
            "role": operation["role"], "purpose": operation["purpose"]}))

    @property
    def reservation(self):
        return loads(self.facts)["reservation"]

    def record_locator(self, locator):
        fixed = loads(self.facts)
        if not isinstance(locator, dict) or not locator.get("session_id"):
            raise ProtocolError("native locator needs its original session identity")
        if (locator.get("role") != fixed["role"] or locator.get("source_sha256") != fixed["source"]["sha256"]
                or locator.get("request_sha256") != self.context.request_sha256):
            raise ProtocolError("native role input does not match its reserved request/source/role")
        return self.context.record_intent(operation_id=self.reservation["operation_id"],
            role=fixed["role"], purpose=fixed["purpose"], source=fixed["source"],
            reservation=self.reservation, submission_key=locator["session_id"],
            native_locator=locator, episode=fixed["episode"])

    def authorize(self):
        fixed = loads(self.facts)
        matches = [item for item in self.context.intents()
                   if item["operation_id"] == self.reservation["operation_id"]]
        if (len(matches) != 1 or any(matches[0][name] != fixed[name]
                for name in ("reservation", "source", "role", "purpose", "episode"))
                or not matches[0]["native_locator"]):
            raise ProtocolError("native call lacks its original persisted operation/locator")
        return self.owner.authorize(self.reservation)

    def remaining_timeout_ms(self, maximum):
        """Use this fixed K3 release's original deadline helper, never reset time."""
        from kernelschema.runtime._local import remaining_timeout_ms
        return remaining_timeout_ms({**self.reservation,
            "timeout_ms": min(maximum, self.reservation["active_ms"])})
