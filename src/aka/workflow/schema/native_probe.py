"""Finite qualification of the original AKA no-tools native transport.

Two owner-accounted calls, no evaluator/GPU or business controller. Original
receipts and input/exec locators are retained; continuation never replaces a
launched operation. Qualification covers transport only, never Task/fidelity.
"""
from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from kernelschema.api import open_run, prepare_run
from kernelschema.artifacts import Publication, durable_write, read_regular
from kernelschema.budget import BudgetDenied
from kernelschema.contracts import ProtocolError, UnsupportedError, digest, jcs, loads, sha256
from kernelschema.runtime import ClaudeProfile, RoleDelivery, AdmissionDenied
from kernelschema.runtime._local import locked, process_alive

from .calls import RoleCalls, Unsettled
from .session import NATIVE_PROFILE, RoleSession, native_agent
from .state import Transactions, invocation_from_document

FORMAT = "aka.native.qualification.v1"
NAMESPACE = "aka-native-qualification"
DEFAULT_SCOPE_TIMEOUT_MS = 120000


def scope_timeout(profile, value=None):
    """A new scope's ceiling; the declared Claude profile remains unchanged."""
    declared = ClaudeProfile(**profile["parameters"]["claude"]).timeout_ms
    if value is None:
        return min(DEFAULT_SCOPE_TIMEOUT_MS, declared)
    if type(value) is not int or not 0 < value <= min(declared, 9007199254740991):
        raise ProtocolError("native probe scope timeout must be a positive safe integer no greater than the declared profile timeout")
    return value


def implementation():
    from . import session, stdin_exec
    from aka.agent import native, provider, runtime
    from aka.agent.backends import base, claude
    from aka.agent.execution import process
    modules = (session, stdin_exec, native, provider, runtime, base, claude, process)
    return {m.__name__: sha256(Path(m.__file__).read_bytes()) for m in modules}


def identity(profile, *, scope_timeout_ms=None):
    from .release import verify_release
    from . import calls, state
    return {"probe": FORMAT, "claude_sha256": digest(profile["parameters"]["claude"]),
        "scope_timeout_ms": scope_timeout(profile, scope_timeout_ms),
        "implementation": implementation(), "ks_manifest": verify_release()["manifest_sha256"],
        "capture": sha256(Path(__file__).read_bytes()),
        "transactions": {m.__name__: sha256(Path(m.__file__).read_bytes()) for m in (calls, state)}}


def validate_capture(bundle, capture, profile, role, *, scope_timeout_ms):
    """Reinterpret original selected artifacts without accessing old private paths."""
    if scope_timeout_ms is None:
        raise ProtocolError("native qualification scope timeout is missing")
    scope_timeout({"parameters": {"claude": profile.as_dict()}}, scope_timeout_ms)
    invocation = invocation_from_document(loads(bundle.read(capture["invocation"])))
    locator = capture["locator"]
    payload = loads(bundle.read(capture["input"]))
    execution = loads(bundle.read(capture["execution"]))
    if (locator["role"] != role or locator["profile"] != NATIVE_PROFILE
            or locator["native_context_resume"] is not False
            or type(locator.get("maximum_timeout_ms")) is not int
            or not 0 < locator["maximum_timeout_ms"] <= min(profile.timeout_ms, scope_timeout_ms)
            or type(locator.get("timeout_ceiling_ms")) is not int
            or locator["timeout_ceiling_ms"] != scope_timeout_ms
            or invocation.session_id != locator["session_id"]
            or sha256(bundle.read(capture["input"])) != locator["input_sha256"]
            or any(payload[k] != locator[k] for k in ("role", "assignment_sha256", "source_sha256"))
            or payload["fixed_assignment"]["request_sha256"] != locator["request_sha256"]
            or digest(payload["fixed_assignment"]) != locator["assignment_sha256"]
            or execution["session_id"] != locator["session_id"]
            or execution["operation_id"] != capture["operation_id"]
            or execution.get("locator_sha256") != digest(locator)
            or type(execution["pid"]) is not int or execution["pid"] <= 0
            or not execution["boot_id"] or not str(execution["start_ticks"]).isdigit()):
        raise ProtocolError("native qualification input/process/role locator mismatch")
    inspector = object.__new__(RoleSession)
    inspector.role = SimpleNamespace(document={"role": role}, assignment_sha256=locator["assignment_sha256"])
    inspector.profile, inspector.workspace = profile, Path(locator["workspace"])
    inspector._session = SimpleNamespace(session_id=locator["session_id"])
    errors, output = inspector._inspect(invocation)
    if errors or output is None or output.status != "ok" or invocation.terminal_usage.total_tokens is None:
        raise UnsupportedError("AKA native permission/output/usage capture incomplete: " + ", ".join(errors))
    return invocation


def references(value):
    return [entry[key] for entry in value["roles"].values() for key in ("invocation", "input", "execution")]


def select_request(request_path, capture_root, out, *, run_root):
    """Author a new immutable AKA selection using captured refs, never a pass flag.

    Existing K6 evaluation parameters and parent account Ref are preserved. The
    caller still supplies that original live authority when executing the child.
    This copies the selected noncredential artifacts, not the capture run/owner.
    """
    from kernelschema.artifacts import Bundle, dependencies, import_bundle
    from kernelschema.materials import validate_request
    from . import backend, agate
    from .session import native_capabilities
    request_path, capture_root, out = Path(request_path).resolve(), Path(capture_root).resolve(), Path(out).resolve()
    if out.exists():
        raise ProtocolError("selection output must be a new directory")
    original = Bundle(request_path.parent)
    request = loads(request_path.read_bytes())
    objects = validate_request(original, request)
    capture_bundle = Bundle(capture_root)
    result = loads(read_regular(capture_root, "native-qualification-result.json")[0])
    reference = result["qualification"]
    if result["status"] != "completed" or reference is None or result["synthetic"] is not False:
        raise UnsupportedError("select an actual completed native capture; synthetic and blocked probes cannot qualify")
    value = loads(capture_bundle.read(reference))
    retained = [reference, *references(value)]
    extra = [{"id": "aka_native_capture_" + str(index), "file": ref, "roles": [],
        "origin": "explicit original AKA transport capture; private host qualification only",
        "license": "private execution evidence; not licensed for redistribution"} for index, ref in enumerate(retained)]
    materials = {**objects["materials"], "entries": sorted([*objects["materials"]["entries"], *extra], key=lambda e: e["id"])}
    # Validate original observations before copying or selecting them.
    agate.native_admission(SimpleNamespace(bundle=capture_bundle, objects={"profile": objects["profile"], "materials": materials}), reference)
    import_bundle(original, out, dependencies(request))
    import_bundle(capture_bundle, out, retained)
    publication = Publication(out)
    profile = loads(jcs(objects["profile"]))
    profile.update(name="aka", capabilities=native_capabilities())
    profile["parameters"].setdefault("aka", {})["services"] = {
        "factory": "aka.workflow.schema.agate:workflow_services",
        "implementation_sha256": sha256(Path(agate.__file__).read_bytes()),
        "parameters": {"native_qualification": reference}}
    request = {**request, "backend": {"name": "aka", "profile": publication.put_json("aka-profile.json", profile)},
        "materials": publication.put_json("aka-materials.json", materials)}
    request_ref = publication.put_json("aka-request.json", request)
    publication.commit([request_ref])
    durable_write(out / "request.json", jcs(request))
    validate_request(Bundle(out), request)
    selection = {"format": "aka.schema.selection.v1", "request": str(out / "request.json"),
        "run_root": str(Path(run_root).resolve()), "backend": {"factory": "aka.workflow.schema.backend:create_workflow",
            "implementation_sha256": sha256(Path(backend.__file__).read_bytes()), "parameters": {}}}
    durable_write(out / "selection.json", jcs(selection))
    return selection


def capture(context, *, continuing=False, agent=None):
    """Use the original RunContext/OwnerAccount; injected agents are synthetic."""
    with locked(context.root / ".runtime/aka-native-probe-lock"):
        return _capture(context, continuing=continuing, agent=agent)


def _capture(context, *, continuing, agent):
    if (context.root / ".runtime/state/aka-fixed-backend.json").exists():
        raise ProtocolError("native qualification requires its own prepared child request, not an admitted business run")
    profile = ClaudeProfile(**context.objects["profile"]["parameters"]["claude"])
    original = context.anchor.get("implementation", {})
    ceiling = original.get("scope_timeout_ms")
    if ceiling is None or original != identity(context.objects["profile"], scope_timeout_ms=ceiling):
        raise ProtocolError("native probe scope/implementation binding missing or changed; retain original source for settlement")
    state_path = context.root / f".runtime/state/{NAMESPACE}.json"
    existing = state_path.exists()
    prior_paths = (".runtime/state/budget.json", ".runtime/owner-account.json",
                   ".runtime/aka-native", ".runtime/terminals", "native-qualification-result.json")
    if not existing and (continuing or context.intents()
            or any((context.root / name).exists() for name in prior_paths)):
        raise ProtocolError("original native probe scope state missing; cannot recreate its binding")
    if existing and context.load_state(NAMESPACE).get("binding") != original:
        raise ProtocolError("original native probe scope binding missing or changed")
    # The exact profile freezes the native USD stopping threshold. Scope and
    # owner limits still bound the finite coder/reviewer transport calls.
    source = context.request["seed"] or context.objects["task"]["starter_source"]
    if context.request["mode"] != "code" or source is None or not context.bundle.source(source):
        raise UnsupportedError("native coder/reviewer probe requires code mode and an explicit nonempty Task source")
    tx = Transactions(context, original, continuing=continuing or existing, namespace=NAMESPACE)
    limits = context.objects["profile"]["native_limits"].get("workflow_limits", {})
    if (context.request["budget"]["limits"]["tokens"] is not None
            or tx.owner.descriptor["limits"]["tokens"] is not None or limits.get("tokens") is not None):
        raise UnsupportedError("native probe cannot enforce finite token budgets")
    injected = agent is not None
    state = tx.state
    if existing:
        probe = state.get("probe")
        if not isinstance(probe, dict) or not {"roles", "qualification", "synthetic"} <= probe.keys():
            raise ProtocolError("original native probe scope records missing")
        if any(op.get("maximum_timeout_ms") != ceiling for op in state["operations"].values()):
            raise ProtocolError("original native probe operation timeout binding missing or changed")
    else:
        probe = state["probe"] = {"roles": {}, "qualification": None, "synthetic": injected}
        tx.save("probe_scope_bound")
    if probe["synthetic"] != injected:
        raise ProtocolError("native probe transport cannot change on continuation")
    own_agent = native_agent(profile) if agent is None else agent
    calls = RoleCalls(tx, own_agent)
    try:
        if state["phase"] != "done":
            tx.owner.begin(context.run_id)
            if "instruction" not in probe:
                publication = Publication(context.root)
                probe["instruction"] = publication.put_bytes("transport-probe.txt", (
                    "Host transport probe only. Do not solve or claim Task, GPU, performance or fidelity qualification. "
                    "For coder return current_source unchanged as candidate_source. For reviewer return a review "
                    "object with transport_probe=true and qualification='transport only; Task and fidelity not tested'. "
                    "Use the original RoleOutput envelope and fixed assignment identity.\n").encode())
                publication.commit([probe["instruction"]])
                tx.save("probe_instruction")
            for role in ("coder", "reviewer"):
                if role in probe["roles"]:
                    continue
                output, _, op, receipt = calls.invoke("native-probe-" + role, role, source,
                    deliveries=(RoleDelivery("native_transport_probe", probe["instruction"], (role,)),),
                    floor=tx.finish_floor(evaluate=False, review=role == "coder"), maximum_timeout_ms=ceiling)
                if not receipt.ok:
                    raise UnsupportedError("original " + role + " probe failed: " + ", ".join(output.permission_errors))
                execution = loads(read_regular(Path(op["locator"]["output_directory"]), "exec-intent.json")[0])
                if process_alive(execution) is not False:
                    raise UnsupportedError("original native process termination is not established")
                publication = Publication(context.root)
                entry = {"operation_id": op["id"], "locator": op["locator"], "invocation": receipt.receipt,
                    "input": publication.put_bytes("native-input.json", read_regular(Path(op["locator"]["output_directory"]), "role-input.json")[0]),
                    "execution": publication.put_bytes("native-exec.json", jcs(execution))}
                publication.commit([entry["input"], entry["execution"]])
                validate_capture(context.bundle, entry, profile, role, scope_timeout_ms=ceiling)
                probe["roles"][role] = entry
                tx.save("probe_role_captured")
            publication = Publication(context.root)
            value = {"format": FORMAT, "profile": NATIVE_PROFILE, "claude_sha256": digest(profile.as_dict()),
                "scope_timeout_ms": ceiling,
                "implementation": implementation(), "synthetic": injected, "roles": probe["roles"],
                "scope": "native transport only; no Task/GPU/fidelity qualification"}
            probe["qualification"] = publication.put_bytes("native-qualification.json", jcs(value))
            publication.commit([probe["qualification"], *references(value)])
            state.update(phase="done", stop_reason="original native transport captured")
            tx.save("probe_complete")
    except (UnsupportedError, BudgetDenied, AdmissionDenied, PermissionError) as error:
        state.update(phase="done", stop_reason=str(error))
        tx.save("probe_stopped")
    except Unsettled as error:
        state["stop_reason"] = str(error)
        tx.save("probe_unsettled")
    except (OSError, ProtocolError) as error:
        state.update(phase="done", stop_reason=str(error))
        tx.save("probe_capture_incomplete")
    finally:
        tx.owner.end(context.run_id)
        if not injected:
            own_agent.close()
    usage_id = digest({"native_probe_usage": context.run_id, "operations": tx.owner.operations(context.run_id)})[7:39]
    usage = tx.owner.usage(context.run_id, Publication(context.root, usage_id))
    value = {"status": "completed" if probe["qualification"] else "blocked",
        "recovery": "settle_only" if usage["unsettled"] else "none", "reason": state["stop_reason"],
        "qualification": probe["qualification"], "synthetic": injected, "usage": usage}
    amounts = []
    for op in state["operations"].values():
        invocation = tx.invocation(op)
        events = []
        for line in invocation.stdout.splitlines() if invocation else ():
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
        totals = [event.get("total_cost_usd") for event in events if isinstance(event, dict) and event.get("type") == "result"]
        amounts.append(str(totals[0]) if len(totals) == 1 and totals[0] is not None else None)
    value["cli_reported_usd"] = str(sum((Decimal(a) for a in amounts), Decimal(0))) if all(a is not None for a in amounts) else None
    durable_write(context.root / "native-qualification-result.json", jcs(value), replace=True)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--request", type=Path)
    selection.add_argument("--continue", dest="continuing", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--scope-timeout-ms", type=int,
        help="new scope's per-call ceiling in ms (default: 120000, narrowed to the declared profile); cannot override --continue")
    args = parser.parse_args(argv)
    if args.continuing:
        if args.out or args.scope_timeout_ms is not None:
            parser.error("--out and --scope-timeout-ms are only for a new request")
        context = open_run("implement", args.continuing)
    else:
        if args.out is None:
            parser.error("--out is required for a new request")
        from kernelschema.artifacts import Bundle
        bundle = Bundle(args.request.resolve().parent)
        request = loads(args.request.read_bytes())
        context = prepare_run("implement", request=args.request, out=args.out,
            backend_identity=identity(bundle.document(request["backend"]["profile"]),
                scope_timeout_ms=args.scope_timeout_ms))
    print(jcs(capture(context, continuing=bool(args.continuing))).decode())


if __name__ == "__main__":
    main()
