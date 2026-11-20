"""Durable fixed-backend transactions over the sole KS account.

This state contains operation locators and policy decisions, never a balance.
All quota, activity and cumulative receipt arithmetic belongs to OwnerAccount.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from threading import RLock
from uuid import uuid4

from kernelschema.artifacts import Publication, durable_write, read_regular
from kernelschema.budget import OwnerAccount, cost
from kernelschema.contracts import ProtocolError, jcs, loads

NAMESPACE = "aka-fixed-backend"
VERSION = "aka.fixed.v1"


def put_once(path, value):
    raw = jcs(value)
    if Path(path).exists():
        if read_regular(Path(path).parent, Path(path).name)[0] != raw:
            raise ProtocolError("immutable AKA transaction changed: " + Path(path).name)
    else:
        durable_write(Path(path), raw)


def invocation_document(invocation):
    return json.loads(json.dumps(asdict(invocation)))


def invocation_from_document(value):
    from aka.contracts.agent import NativeInvocation, TokenUsage, NormalizedAgentEvent, AgentRuntimeCapabilities
    value = dict(value)
    value["terminal_usage"] = TokenUsage(**value["terminal_usage"])
    value["capabilities"] = AgentRuntimeCapabilities(**value["capabilities"])
    value["events"] = tuple(NormalizedAgentEvent(**{**item, "usage": TokenUsage(**item["usage"]) if item["usage"] else None})
                            for item in value["events"])
    value["observation_errors"] = tuple(value["observation_errors"])
    return NativeInvocation(**value)


class Transactions:
    def __init__(self, context, binding, *, continuing=False, checkpoint=None, namespace=NAMESPACE):
        self.context = context
        self.namespace = namespace
        self._lock = RLock()
        self.owner = OwnerAccount.for_context(context, create=not continuing)
        self.checkpoint = checkpoint or (lambda boundary: None)
        path = context.root / f".runtime/state/{namespace}.json"
        if path.exists():
            self.state = context.load_state(namespace)
            if self.state["binding"] != binding or self.state["version"] != VERSION:
                raise ProtocolError("original AKA backend/native configuration changed")
        else:
            if context.intents() or self.owner.operations(context.run_id):
                raise ProtocolError("lost AKA state cannot acquire new operations")
            self.state = dict(version=VERSION, binding=binding, operations={}, episodes={},
                latest=context.request["seed"] or context.objects["task"]["starter_source"], qualified=None,
                assessments=[], feedback=[], status="partial", recovery="none", stop_reason="initializing",
                phase="running", finished=False, result=None, diagnostics=[])
            self.save("created")
        self.validate()

    def save(self, boundary):
        with self._lock:
            self.context.save_state(self.namespace, self.state)
        self.checkpoint(boundary)

    def validate(self):
        owned = self.owner.operations(self.context.run_id)
        declared = {op["id"]: op for op in self.state["operations"].values()}
        if not set(owned) <= declared.keys():
            raise ProtocolError("owner operation missing from retained AKA transactions")
        for op in declared.values():
            if op.get("reservation") and owned.get(op["id"], {}).get("reservation") != op["reservation"]:
                raise ProtocolError("AKA operation differs from its sole owner")
        for intent in self.context.intents():
            if intent["operation_id"] not in declared or intent["reservation"] != owned[intent["operation_id"]]["reservation"]:
                raise ProtocolError("original AKA intent/owner binding lost")

    def plan(self, key, *, role, source, episode=None, purpose=None):
        expected = dict(role=role, source=source, episode=episode, purpose=purpose or key)
        found = self.state["operations"].get(key)
        if found is not None:
            if any(found[k] != v for k, v in expected.items()):
                raise ProtocolError("original operation key changed subject")
            return found
        op = dict(expected, id="aka_" + uuid4().hex, publication_id=uuid4().hex,
                  reservation=None, terminal=None, locator=None)
        self.state["operations"][key] = op
        self.save("operation_planned")
        return op

    def reserve(self, op, *, floor=None, jobs=1):
        if op["reservation"] is None:
            minimum = self.context.objects["profile"]["parameters"].get("minimum_active_ms",
                {"coder": 1000, "evaluator": 1000, "reviewer": 1000})
            if set(minimum) != {"coder", "evaluator", "reviewer"} or any(type(n) is not int or n <= 0 for n in minimum.values()):
                raise ProtocolError("positive coder/evaluator/reviewer minimum_active_ms required")
            role = op["role"]
            op["reservation"] = self.owner.reserve(self.context.run_id, op["id"], role=role,
                purpose=op["purpose"], amount=cost(role=role, calls=int(role != "evaluator"),
                    evaluator_jobs=jobs if role == "evaluator" else 0, active_ms=minimum[role]),
                floor=floor or cost(), native_limits=self.context.objects["profile"]["native_limits"].get("workflow_limits"))
            self.save("operation_reserved")
        return op["reservation"]

    def finish_floor(self, *, evaluate=True, review=True):
        minimum = self.context.objects["profile"]["parameters"].get("minimum_active_ms",
            {"coder": 1000, "evaluator": 1000, "reviewer": 1000})
        return cost(role="reviewer", calls=int(review), evaluator_jobs=int(evaluate),
                    active_ms=minimum["evaluator"] * int(evaluate) + minimum["reviewer"] * int(review))

    def intent(self, op, locator, *, key=None):
        if op["locator"] is not None and op["locator"] != locator:
            raise ProtocolError("original native locator changed")
        op["locator"] = locator
        self.context.record_intent(operation_id=op["id"], role=op["role"], purpose=op["purpose"],
            source=op["source"], reservation=op["reservation"], submission_key=key or op["id"],
            native_locator=locator, episode=op["episode"])
        self.save("operation_intent")

    def absorb(self, op, row, *, terminated=True, evidence=()):
        # Native terminal is evidence about this operation only, never its siblings.
        if row["operation_id"] != op["id"] or row["role"] != op["role"]:
            raise ProtocolError("receipt belongs to another original operation")
        self.owner.absorb(self.context.run_id, row, terminated=terminated)
        facts = {"usage": row, "terminated": terminated, "evidence": list(evidence)}
        if terminated:
            if not (self.context.root / ".runtime/terminals" / (op["id"] + ".json")).exists():
                self.context.record_terminal(op["id"], facts)
            history = op.setdefault("settlements", [])
            if facts not in history:
                history.append(facts)
            op["terminal"] = facts
        self.save("operation_absorbed")

    def unknown(self, op, reason):
        self.owner.mark_unknown(op["id"], reason)
        self.state.update(status="blocked", recovery="settle_only", stop_reason=reason)
        self.save("operation_unknown")

    def retain_invocation(self, op, invocation):
        path = self.context.root / ".runtime/aka-native" / op["id"] / "invocation.json"
        put_once(path, {"operation_id": op["id"], "invocation": invocation_document(invocation)})
        self.checkpoint("native_invocation_retained")

    def invocation(self, op):
        path = self.context.root / ".runtime/aka-native" / op["id"] / "invocation.json"
        if not path.exists():
            return None
        row = loads(read_regular(path.parent, path.name)[0])
        if row["operation_id"] != op["id"]:
            raise ProtocolError("retained native invocation belongs to another operation")
        return invocation_from_document(row["invocation"])

    def absorb_invocation(self, op, invocation):
        publication = Publication(self.context.root, op["publication_id"])
        reference = publication.put_bytes("native-invocation.json", jcs(invocation_document(invocation)))
        publication.commit([reference], subjects=[self.context.request_ref, op["source"]])
        row = dict(producer="aka.native:" + self.context.run_id, operation_id=op["id"], role=op["role"],
            purpose=op["purpose"], receipt_id=reference["sha256"], sequence=1, receipt=reference,
            state="settled", calls=1, evaluator_jobs=0, tokens=invocation.terminal_usage.total_tokens, gpu_ms=0)
        self.absorb(op, row, evidence=[reference])
        return reference
