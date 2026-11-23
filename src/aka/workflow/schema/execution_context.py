"""Small host-owned projection and exact Contract correction authority.

The controller supplies frozen Task/policy bytes and a verifier for retained
native evidence. That verifier must establish ownership (Task, target, policy,
route/environment applicability) and distinguish observations from declarations
before returning the fixed projection below. This module checks the original
byte digest, exact bindings, field types and provenance; it does not discover an
environment, infer architecture from a device name, or parse Task prose.

Task inspection is another trusted controller callback over the *complete*
public Task. Missing inspection means unknown, never permission to correct.
Callbacks/constructors are host APIs, not model tools. Python callers already
own the Task and policy; this is not a sandbox against a malicious controller.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Callable, Mapping

from .materials import DesignRef, _json, read_design
from .language.yaml_source import read_source_bytes
from aka.task.source.inspection import framework_key

PREFIX = "root.CONTRACT.Hardware_specification."
CORRECTABLE = (PREFIX + "Architecture", PREFIX + "Stack.Language", PREFIX + "Stack.Toolchain")
SOURCES = frozenset({"observed", "configured", "declared", "cache", "unknown"})
VERSION_TOKEN = r"[0-9]{1,4}(?:\.[0-9]{1,4}){1,3}(?:[a-z][a-z0-9.]{0,16})?(?:\+(?:cu[0-9]{2,4}|cpu))?"
# Exact representations, not substring tests against authored prose.
STACKS = {"triton": ("Triton", "Triton"), "cuda": ("CUDA C++", "nvcc"),
          "pytorch": ("PyTorch", "PyTorch"), "torch": ("PyTorch", "PyTorch"),
          "cutlass": ("CUTLASS", "CUTLASS"), "cutedsl": ("CuTe DSL", "CuTe DSL"),
          "tilelang": ("TileLang", "TileLang"), "flydsl": ("FlyDSL", "FlyDSL"),
          "gluon": ("Gluon", "Gluon")}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hash(value):
    return sha256(_json(value))


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("execution evidence requires a SHA256 digest")
    return value


def execution_binding(*, prepared_id, public_task, target_hardware, native_policy):
    """Binding used by the native evidence verifier; policy matches state.digest."""
    if not all(isinstance(item, str) and item.strip() for item in (prepared_id, public_task, target_hardware)):
        raise ValueError("execution context requires frozen prepared/Task/hardware values")
    if not isinstance(native_policy, Mapping):
        raise ValueError("execution context requires the frozen native policy mapping")
    return {"prepared_id_sha256": sha256(prepared_id.encode()),
            "public_task_sha256": sha256(public_task.encode()),
            "target_hardware_sha256": sha256(target_hardware.encode()),
            "native_policy_sha256": sha256(json.dumps(native_policy, sort_keys=True,
                ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())}


def _fact(value, *, field, evidence_sha256, framework):
    fields = {"value", "source", "framework"} if field == "toolchain" else {"value", "source"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("execution fact has missing or unexpected fields")
    if field == "toolchain" and (not isinstance(value["framework"], str)
            or framework_key(value["framework"]) != framework):
        raise ValueError("toolchain evidence belongs to another selected framework")
    source, token = value["source"], value["value"]
    if not isinstance(source, str) or source not in SOURCES:
        raise ValueError("invalid execution fact provenance")
    if source == "unknown":
        if token is not None:
            raise ValueError("unknown execution fact cannot claim a value")
    else:
        pattern = (r"sm_[0-9]{2,3}[a-z]?|gfx[0-9a-f]{3,5}" if field == "architecture"
                   else VERSION_TOKEN)
        if not isinstance(token, str) or re.fullmatch(pattern, token) is None:
            raise ValueError("execution fact is not a supported public architecture/version token")
    return {"value": token, "source": source, "evidence_sha256": evidence_sha256}


def _canonical(fact, name):
    if fact["source"] == "unknown":
        # Required text can state an epistemic limit without inventing an ISA.
        # This does not prove support for any mechanism used elsewhere. Keep
        # conservative designs available under the original full checker.
        return "Unspecified architecture" if name == "architecture" else "(version unspecified)"
    text = fact["value"]
    return text if fact["source"] == "observed" else text + " (" + fact["source"] + ")"


def _check_constraints(constraints, canonical):
    if not isinstance(constraints, dict) or set(constraints) != set(CORRECTABLE):
        raise ValueError("Task inspection must cover exactly the permitted hardware leaves")
    for path, constraint in constraints.items():
        if (not isinstance(constraint, dict) or set(constraint) != {"status", "value"}
                or not isinstance(constraint["status"], str)
                or constraint["status"] not in {"explicit", "unspecified", "unknown"}):
            raise ValueError("invalid Task hardware constraint inspection")
        if constraint["status"] == "explicit":
            if constraint["value"] != canonical[path]:
                raise ValueError("explicit public Task constraint conflicts with native execution context")
        elif constraint["value"] is not None:
            raise ValueError("unspecified/unknown Task constraint cannot claim a value")


@dataclass(frozen=True, init=False)
class ExecutionContext:
    """Immutable host-created bytes; no caller/model mapping is accepted by Planner."""
    _raw: bytes

    @classmethod
    def _freeze(cls, document):
        result = object.__new__(cls)
        object.__setattr__(result, "_raw", _json(document))
        return result

    @property
    def sha256(self):
        return sha256(self._raw)

    def as_dict(self):
        return json.loads(self._raw)

    def as_bytes(self):
        return self._raw

    def validate_request(self, prepared_id, public_task, target_hardware):
        doc = self.as_dict()
        binding = doc["binding"]
        expected = {"prepared_id_sha256": sha256(prepared_id.encode()),
                    "public_task_sha256": sha256(public_task.encode()),
                    "target_hardware_sha256": sha256(target_hardware.encode())}
        if (any(binding[name] != value for name, value in expected.items())
                or doc["canonical_hardware"][PREFIX + "Device"] != target_hardware.strip()):
            raise ValueError("execution context belongs to another prepared Task or target")


def build_execution_context(*, prepared_id: str, public_task: str, target_hardware: str,
                            native_policy: Mapping, evidence=None,
                            verify_evidence: Callable | None = None,
                            inspect_task: Callable | None = None) -> ExecutionContext:
    """Verify then freeze the only supported public fields.

    evidence is one optional (original_bytes, expected_original_sha256) pair.
    verify_evidence(bytes) must return exactly {binding, architecture, toolchain},
    with binding from execution_binding and facts {value: token|null, source:
    observed|configured|declared|cache|unknown}; toolchain also requires framework
    matching the selected policy (versions cannot cross software stacks).
    It must check native ownership
    and applicability using retained owner data, not campaign.arch or prompt text.

    inspect_task(public_task) returns all three CORRECTABLE paths, each exactly
    {status: explicit|unspecified|unknown, value: canonical_string|null}.
    'unspecified' attests complete Task review, not a failed keyword search.
    Explicit values must agree with native facts; conflicts never change a GPU
    or public requirement. Nothing from raw evidence/policy is exported except
    the allowlisted tokens and hashes. Verifier exceptions are sanitized.
    """
    binding = execution_binding(prepared_id=prepared_id, public_task=public_task,
        target_hardware=target_hardware, native_policy=native_policy)
    selected = native_policy.get("framework")
    framework = framework_key(selected) if isinstance(selected, str) else None
    if framework not in STACKS:
        raise ValueError("execution context has an unsupported selected framework")
    if native_policy.get("sandbox_hardware") != target_hardware:
        raise ValueError("native policy and frozen target hardware conflict")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._()+/-]{0,99}", target_hardware.strip()) is None:
        raise ValueError("execution context requires a bounded public device name")
    facts = {name: {"value": None, "source": "unknown", "evidence_sha256": None}
             for name in ("architecture", "toolchain")}
    if evidence is not None:
        raw, expected = evidence
        _digest(expected)
        if not isinstance(raw, bytes) or sha256(raw) != expected or not callable(verify_evidence):
            raise ValueError("execution evidence needs original bytes, matching digest and host verifier")
        try:
            projection = verify_evidence(raw)
        except Exception:
            raise ValueError("native execution evidence ownership verification failed") from None
        if not isinstance(projection, dict) or set(projection) != {"binding", "architecture", "toolchain"}:
            raise ValueError("native verifier returned fields outside the execution allowlist")
        if projection["binding"] != binding:
            raise ValueError("native evidence belongs to another prepared Task, target or policy")
        facts = {name: _fact(projection[name], field=name, evidence_sha256=expected, framework=framework)
                 for name in facts}
    constraints = {path: {"status": "unknown", "value": None} for path in CORRECTABLE}
    if inspect_task is not None:
        if not callable(inspect_task):
            raise ValueError("Task constraint inspection requires a host verifier")
        try:
            constraints = inspect_task(public_task)
        except Exception:
            raise ValueError("public Task hardware constraint inspection failed") from None
    canonical = {PREFIX + "Device": target_hardware.strip(),
                 CORRECTABLE[0]: _canonical(facts["architecture"], "architecture"),
                 CORRECTABLE[1]: STACKS[framework][0],
                 CORRECTABLE[2]: STACKS[framework][1] + " " + _canonical(facts["toolchain"], "toolchain")}
    _check_constraints(constraints, canonical)
    return ExecutionContext._freeze({"format": "aka.schema.execution-context.v1", "binding": binding,
        "selected_framework": {"value": framework, "source": "configured",
                               "evidence_sha256": binding["native_policy_sha256"]},
        **facts, "task_constraints": constraints, "canonical_hardware": canonical})


def restore_execution_context(raw: bytes, *, expected_sha256: str) -> ExecutionContext:
    """Replay from host-retained execution.json and its ORIGINAL intent hash.

    The expected hash must come from the controller's frozen operation, never
    from a model's claimed digest. No environment acquisition or Task/evidence
    reinterpretation runs during restore. This is the same trusted host input
    boundary as restoring other operation inputs, not an evidence constructor.
    """
    _digest(expected_sha256)
    if not isinstance(raw, bytes) or sha256(raw) != expected_sha256:
        raise ValueError("frozen execution context differs from its original input hash")
    doc = json.loads(raw)
    if (not isinstance(doc, dict) or set(doc) - {"legacy_device_label"} != {"format", "binding", "selected_framework",
            "architecture", "toolchain", "task_constraints", "canonical_hardware"}
            or doc["format"] != "aka.schema.execution-context.v1"):
        raise ValueError("invalid frozen execution context schema")
    if set(doc["binding"]) != {"prepared_id_sha256", "public_task_sha256",
                               "target_hardware_sha256", "native_policy_sha256"}:
        raise ValueError("invalid frozen execution binding")
    for digest in doc["binding"].values():
        _digest(digest)
    selection = doc["selected_framework"]
    if (set(selection) != {"value", "source", "evidence_sha256"}
            or selection["value"] not in STACKS or selection["source"] != "configured"
            or selection["evidence_sha256"] != doc["binding"]["native_policy_sha256"]):
        raise ValueError("invalid frozen framework selection")
    framework = selection["value"]
    for name in ("architecture", "toolchain"):
        fact = doc[name]
        if set(fact) != {"value", "source", "evidence_sha256"}:
            raise ValueError("invalid frozen execution fact")
        if fact["source"] != "unknown" or fact["evidence_sha256"] is not None:
            _digest(fact["evidence_sha256"])
        fields = {"value": fact["value"], "source": fact["source"]}
        if name == "toolchain":
            fields["framework"] = framework
        _fact(fields, field=name, evidence_sha256=fact["evidence_sha256"], framework=framework)
    canonical = doc["canonical_hardware"]
    if (set(canonical) != {PREFIX + "Device", *CORRECTABLE}
            or not isinstance(canonical[PREFIX + "Device"], str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._()+/-]{0,99}", canonical[PREFIX + "Device"]) is None
            or canonical[CORRECTABLE[0]] != _canonical(doc["architecture"], "architecture")
            or canonical[CORRECTABLE[1]] != STACKS[framework][0]
            or canonical[CORRECTABLE[2]] != STACKS[framework][1] + " " + _canonical(doc["toolchain"], "toolchain")):
        raise ValueError("invalid frozen canonical hardware leaves")
    _check_constraints(doc["task_constraints"], canonical)
    if "legacy_device_label" in doc:
        label = doc["legacy_device_label"]
        if (not isinstance(label, dict) or set(label) != {"parent_identity", "parent_operation_sha256", "value"}
                or not isinstance(label["value"], str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._()+/-]{0,99}", label["value"]) is None
                or label["value"] == canonical[PREFIX + "Device"]):
            raise ValueError("invalid frozen legacy Device declaration")
        _digest(label["parent_identity"])
        _digest(label["parent_operation_sha256"])
    context = ExecutionContext._freeze(doc)
    if context.as_bytes() != raw:
        raise ValueError("frozen execution context is not canonical host input")
    return context


def context_input(context, *, prepared_id, public_task, target_hardware):
    if not isinstance(context, ExecutionContext):
        raise ValueError("execution context must come from the host verified-source constructor")
    context.validate_request(prepared_id, public_task, target_hardware)
    return context.as_bytes()


def _leaf(contract, path):
    value = contract
    try:
        for name in path.removeprefix("root.CONTRACT.").split("."):
            value = value[name]
    except (KeyError, TypeError):
        raise ValueError("execution context requires the complete hardware leaf: " + path) from None
    return value


def check_context_hardware(checked, context, *, parent=None):
    if context is None:
        return
    contract = checked.resolved.raw_task["root"]["CONTRACT"]
    canonical = context.as_dict()["canonical_hardware"]
    canonical[PREFIX + "Device"] = _device_for_design(parent, context)
    for path, expected in canonical.items():
        if _leaf(contract, path) != expected:
            raise ValueError("hardware leaf must equal host canonical value: " + path + " = " + expected)


def _origin_inputs(artifact, *, required=False):
    """Read the existing Planner operation, not a second authority ledger.

    The first implementation intentionally supports corrections of Planner's
    own adjacent publications only. Imports/extractions need an owner adapter,
    not an inferred permission from a matching prepared id.
    """
    path = artifact.ref.directory.parent / "intent.json"
    if not path.exists():
        if required:
            raise ValueError("correction requires the original Planner operation inputs")
        return None
    raw = read_source_bytes(path)
    intent = json.loads(raw)
    if (intent.get("format") != "aka.schema.role-operation.v1"
            or intent.get("mode") not in {"design", "optimize"}
            or intent.get("operation_id") != artifact.manifest["operation_id"]):
        if required:
            raise ValueError("correction origin is not the exact parent Planner operation")
        return None
    return intent["inputs"], sha256(raw)


def retain_legacy_device_label(context, *, parent, label, target_hardware):
    """Trusted Task owner attests this exact old label denotes the frozen target.

    This is an explicit declaration, never alias inference or model evidence.
    Only this publication and its unchanged-Device descendants may retain it.
    The execution target and canonical Device remain unchanged.
    """
    doc = context.as_dict()
    if (sha256(target_hardware.encode()) != doc["binding"]["target_hardware_sha256"]
            or any(row["status"] == "unknown" for row in doc["task_constraints"].values())):
        raise ValueError("legacy Device declaration requires the exact target and complete Task review")
    artifact = read_design(parent)
    doc["legacy_device_label"] = {"parent_identity": parent.identity,
        "parent_operation_sha256": _origin_inputs(artifact, required=True)[1], "value": label}
    result = ExecutionContext._freeze(doc)
    result = restore_execution_context(result.as_bytes(), expected_sha256=result.sha256)
    _device_for_design(artifact, result)
    return result


def _device_for_design(artifact, context):
    doc = context.as_dict()
    label = doc.get("legacy_device_label")
    if label is None:
        return doc["canonical_hardware"][PREFIX + "Device"]
    binding = doc["binding"]
    while artifact is not None:
        inputs, operation_sha256 = _origin_inputs(artifact, required=True)
        if (sha256(artifact.manifest["prepared_id"].encode()) != binding["prepared_id_sha256"]
                or inputs["task/public.md"] != binding["public_task_sha256"]
                or inputs["task/hardware.md"] != binding["target_hardware_sha256"]
                or _leaf(artifact.checked.resolved.raw_task["root"]["CONTRACT"], PREFIX + "Device") != label["value"]):
            raise ValueError("legacy Device declaration differs from its original Task, target or Device")
        original = _publication_context(artifact, (inputs, operation_sha256))
        if original is not None and original.as_dict()["binding"] != binding:
            raise ValueError("legacy Device declaration cannot change native policy")
        if artifact.ref.identity == label["parent_identity"]:
            if operation_sha256 != label["parent_operation_sha256"]:
                raise ValueError("legacy Device declaration origin changed")
            return label["value"]
        parent = artifact.manifest["parent"]
        previous = read_design(DesignRef.from_dict(parent)) if parent else None
        if previous is not None and previous.manifest["version"] >= artifact.manifest["version"]:
            raise ValueError("legacy Device lineage must advance publication version")
        artifact = previous
    raise ValueError("legacy Device declaration requires its reviewed publication lineage")


def validate_design_execution_context(artifact, context):
    """New Coder/reviewer inputs must retain the publication's context digest."""
    if context is not None:
        _device_for_design(artifact, context)
    origin = _origin_inputs(artifact)
    if origin is None:
        return
    inputs, _ = origin
    expected = inputs.get("task/execution.json")
    if expected is not None and (context is None or context.sha256 != expected):
        raise ValueError("design execution context differs from its frozen Planner publication")
    if context is not None:
        binding = context.as_dict()["binding"]
        if (binding["public_task_sha256"] != inputs["task/public.md"]
                or binding["target_hardware_sha256"] != inputs["task/hardware.md"]):
            raise ValueError("design execution context belongs to another public Task or target")


def validate_successor_execution_context(artifact, context, *, public_task, target_hardware):
    """Every optimize successor retains external bindings, including plain rewrites.

    Facts may change in a new operation. A context-bound parent cannot be
    downgraded to the legacy no-context path or rebound to another native policy.
    """
    if context is not None:
        _device_for_design(artifact, context)
    origin = _origin_inputs(artifact)
    if origin is None:
        return
    inputs, _ = origin
    if (sha256(public_task.encode()) != inputs["task/public.md"]
            or sha256(target_hardware.encode()) != inputs["task/hardware.md"]):
        raise ValueError("optimization must retain the parent's public Task and target")
    expected = inputs.get("task/execution.json")
    if expected is None:
        return
    if context is None:
        raise ValueError("optimization cannot discard the parent's frozen execution context")
    original = _publication_context(artifact, origin)
    if original.as_dict()["binding"] != context.as_dict()["binding"]:
        raise ValueError("optimization cannot change the parent's Task or native policy binding")


def publication_execution_context(artifact):
    """Restore a publication's original input, never refresh it from planning facts."""
    return _publication_context(artifact, _origin_inputs(artifact))


def _publication_context(artifact, origin):
    expected = origin[0].get("task/execution.json") if origin is not None else None
    if expected is None:
        return None
    path = artifact.ref.directory.parent / "attempts/1/workspace/inputs/task/execution.json"
    return restore_execution_context(read_source_bytes(path), expected_sha256=expected)


@dataclass(frozen=True)
class ContractCorrection:
    """Exact host authority; Planner recomputes it from context and parent."""
    context_sha256: str
    parent_identity: str
    parent_closure_digest: str
    parent_operation_sha256: str
    # path, old JSON value hash, new JSON value hash, original evidence digest.
    leaves: tuple[tuple[str, str, str, str], ...]

    def as_dict(self):
        return {"format": "aka.schema.contract-correction.v1", "context_sha256": self.context_sha256,
                "parent_identity": self.parent_identity, "parent_closure_digest": self.parent_closure_digest,
                "parent_operation_sha256": self.parent_operation_sha256,
                "leaves": [{"path": path, "old_sha256": old, "new_sha256": new,
                            "evidence_sha256": evidence} for path, old, new, evidence in self.leaves]}

    @property
    def sha256(self):
        return _hash(self.as_dict())


def authorize_contract_correction(context: ExecutionContext, parent: DesignRef) -> ContractCorrection:
    """Only prove mismatching reviewed leaves; no whole-hardware or model grant.

    Observed arch/toolchain evidence is required for correcting those facts.
    Configuration/declaration/cache provenance is preserved for initial designs,
    but cannot turn a legacy assertion into a claimed runtime observation.
    """
    if not isinstance(context, ExecutionContext):
        raise ValueError("correction requires a host verified execution context")
    artifact = read_design(parent)
    doc = context.as_dict()
    if sha256(artifact.manifest["prepared_id"].encode()) != doc["binding"]["prepared_id_sha256"]:
        raise ValueError("correction parent belongs to another prepared Task")
    origin = _origin_inputs(artifact, required=True)
    inputs, operation_sha256 = origin
    if (inputs["task/public.md"] != doc["binding"]["public_task_sha256"]
            or inputs["task/hardware.md"] != doc["binding"]["target_hardware_sha256"]):
        raise ValueError("correction must retain the parent's original public Task and target")
    original = _publication_context(artifact, origin)
    if original is not None and original.as_dict()["binding"] != doc["binding"]:
        raise ValueError("correction cannot change the parent's Task or native policy")
    contract = artifact.checked.resolved.raw_task["root"]["CONTRACT"]
    if _leaf(contract, PREFIX + "Device") != _device_for_design(artifact, context):
        raise ValueError("correction cannot replace the frozen Device")
    changes = []
    for path, name in zip(CORRECTABLE, ("architecture", "selected_framework", "toolchain")):
        old, new = _leaf(contract, path), doc["canonical_hardware"][path]
        if old == new:
            continue
        if doc["task_constraints"][path]["status"] == "unknown":
            raise ValueError("correction requires complete host inspection of the public Task")
        fact = doc[name]
        if name != "selected_framework" and fact["source"] != "observed":
            raise ValueError("runtime fact correction requires applicable observed evidence")
        changes.append((path, _hash(old), _hash(new), fact["evidence_sha256"]))
    if not changes:
        raise ValueError("correction proof needs a mismatching permitted hardware leaf")
    return ContractCorrection(context.sha256, parent.identity, artifact.manifest["closure_digest"],
                              operation_sha256, tuple(changes))


def validate_contract_correction(proof, context, parent):
    if not isinstance(proof, ContractCorrection) or parent is None:
        raise ValueError("Contract correction requires an exact host proof and parent")
    if proof != authorize_contract_correction(context, parent):
        raise ValueError("Contract correction proof differs from its context, parent or exact leaves")


def check_contract_change(previous, checked, *, context, proof):
    old = previous.checked.resolved.raw_task["root"]["CONTRACT"]
    new = checked.resolved.raw_task["root"]["CONTRACT"]
    if proof is None:
        if old != new:
            raise ValueError("optimization cannot change the frozen Contract")
        return False
    # Copy only these exact three possible leaves. Algorithm, ABI, storage,
    # mutation, Device, Wiki, Template, Execution and Feature remain identical.
    expected = json.loads(json.dumps(old))
    for path, _, _, _ in proof.leaves:
        cursor = expected
        parts = path.removeprefix("root.CONTRACT.").split(".")
        for name in parts[:-1]:
            cursor = cursor[name]
        cursor[parts[-1]] = context.as_dict()["canonical_hardware"][path]
    if new != expected:
        raise ValueError("Contract differs outside the exact authorized hardware correction")
    return True
