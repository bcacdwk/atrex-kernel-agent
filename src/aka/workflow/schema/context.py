"""Fixed role input and bounded readers over the original KS Template API."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path

from aka.contracts.agent import AgentAccess, AgentMount, AgentToolBinding
from kernelschema.artifacts import durable_write, read_regular
from kernelschema.contracts import ProtocolError, digest, jcs, loads
from kernelschema.runtime import prepare_role_input
from kernelschema import template_api


@dataclass(frozen=True)
class RoleContext:
    """A serialized grant, not a bundle-root filesystem grant.

    The original K3 input builder verifies Task/acceptance, R, Rules, scope and
    Materials. Its assignment digest stays fixed across explicit current C.
    Mutable dictionaries supplied by callers are never retained as authority.
    """
    run: object
    payload: bytes
    grants: bytes

    @classmethod
    def prepare(cls, run, role, *, source=None, source_roles=("coder", "reviewer"), deliveries=()):
        value = prepare_role_input(run, role, source=source,
                                   source_roles=source_roles, deliveries=deliveries)
        from dataclasses import asdict
        grant = {"role": role, "source": source, "source_roles": list(source_roles),
                 "deliveries": [{**asdict(item), "roles": list(item.roles)} for item in deliveries]}
        return cls(run, jcs(value), jcs(grant))

    @property
    def document(self):
        return loads(self.payload)

    @property
    def assignment_sha256(self):
        return self.document["assignment_sha256"]

    def validate(self):
        document = self.document
        if (document["fixed_assignment"]["request_sha256"] != self.run.request_sha256
                or digest(document["fixed_assignment"]) != document["assignment_sha256"]
                or self.run.bundle.document(self.run.request_ref) != self.run.request):
            raise ProtocolError("fixed role request/assignment changed")
        # Validate all selected immutable objects without broadening the grant.
        from kernelschema.materials import validate_request
        if validate_request(self.run.bundle, self.run.request) != self.run.objects:
            raise ProtocolError("fixed role objects changed")
        from kernelschema.runtime import RoleDelivery
        grant = loads(self.grants)
        rebuilt = prepare_role_input(self.run, grant["role"], source=grant["source"],
            source_roles=tuple(grant["source_roles"]),
            deliveries=tuple(RoleDelivery(**item) for item in grant["deliveries"]))
        if jcs(rebuilt) != self.payload:
            raise ProtocolError("role input differs from its actual Materials/source grant")
        return document

    def retain(self, directory):
        directory = Path(directory)
        self.validate()
        for name, raw in (("role-input.json", self.payload), ("role-grants.json", self.grants)):
            path = directory / name
            if path.exists():
                if read_regular(directory, name)[0] != raw:
                    raise ProtocolError("retained role input cannot be replaced")
            else:
                durable_write(path, raw)
        return directory / "role-input.json"

    @classmethod
    def reopen(cls, run, directory, *, expected_sha256):
        from kernelschema.contracts import sha256
        raw = read_regular(Path(directory), "role-input.json")[0]
        if sha256(raw) != expected_sha256:
            raise ProtocolError("retained role input identity changed")
        result = cls(run, raw, read_regular(Path(directory), "role-grants.json")[0])
        result.validate()
        return result

    @contextmanager
    def tools(self, agent, workspace, *, deadline=None):
        workspace = Path(workspace).resolve(strict=True)
        access = AgentAccess((AgentMount(workspace, workspace, False, "Schema role scope"),))
        with agent.tool_session(workspace, access=access,
                bindings=(AgentToolBinding("schema", RoleReader(self, workspace)),), deadline=deadline) as grant:
            yield grant


class RoleReader:
    """Read selected IDs and check the fixed R. No path, write or execution API."""
    def __init__(self, role, workspace):
        self.role, self.workspace = role, Path(workspace)

    def handle(self, payload, context):
        if context.workspace != self.workspace or context.cancel_event.is_set():
            raise PermissionError("Schema tool scope is unavailable")
        request = loads(payload)
        if not isinstance(request, dict):
            raise PermissionError("Schema tool needs an explicit operation")
        document = self.role.validate()
        operation = request.get("op")
        if operation == "read" and set(request) == {"op", "id"}:
            items = {item["id"]: item for item in document["materials"]}
            if request["id"] not in items:
                raise PermissionError("material ID is outside this role grant")
            value = items[request["id"]]
        elif operation in {"effective", "check"}:
            schema = self.role.run.objects.get("schema")
            if schema is None:
                raise ProtocolError("this assignment has no fixed Schema")
            if operation == "effective" and set(request) == {"op", "node"}:
                value = template_api.effective_document(self.role.run.bundle, schema, request["node"])
            elif operation == "check" and set(request) == {"op"}:
                template_api.resolve_bundle(self.role.run.bundle, schema, validate_structure=True)
                value = {"structure": "pass", "semantics": "not_assessed",
                         "schema_sha256": self.role.run.request["schema"]["sha256"]}
            else:
                raise PermissionError("Schema reader does not accept path or mutation options")
        else:
            raise PermissionError("operation is outside the read/effective/check grant")
        return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
