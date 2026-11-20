"""Public qualification projection, independent of native incumbent promotion."""
from kernelschema.artifacts import Publication
from kernelschema.contracts import PROTOCOL, ProtocolError, digest

CHECKS = ("coverage", "build", "correctness", "fidelity", "performance")


def assessment(state, source):
    found = next((item for item in state["assessments"] if item["source_sha256"] == source["sha256"]), None)
    if found is None:
        found = {"source_sha256": source["sha256"], "task_uncovered": [], "performance_goal": "unknown",
                 **{name: {"status": "not_run", "receipt": None, "reason": "not evaluated"} for name in CHECKS}}
        state["assessments"].append(found)
    return found


def apply_report(state, source, reference, report, names):
    current = assessment(state, source)
    if report["subject"]["source_sha256"] != source["sha256"]:
        raise ProtocolError("qualification report binds a different current Source")
    for name in names:
        check = report[name]
        current[name] = {"status": check["status"], "receipt": reference if check["status"] != "not_run" else None,
                         "reason": check.get("reason", "current fixed Schema independent review")}
    if "coverage" in names:
        current["task_uncovered"] = report["coverage"]["task_uncovered"]
    if "performance" in names:
        goal = report.get("extensions", {}).get("information", {}).get("performance_goal", "unknown")
        current["performance_goal"] = goal if goal in ("met", "unmet", "unknown") else "unknown"
    return current


def qualifies(value):
    return not value["task_uncovered"] and all(value[name]["status"] == "pass" for name in CHECKS)


def result(tx, *, native_incumbent=None):
    context, state = tx.context, tx.state
    owned = tx.owner.operations(context.run_id)
    revision = digest({key: {k: row[k] for k in ("latest", "terminated", "released")}
                       for key, row in owned.items()})
    if state.get("result") is not None and state.get("usage_revision") == revision and not tx.owner.fences():
        return state["result"]
    usage = tx.owner.usage(context.run_id, Publication(context.root))
    if usage["unsettled"]:
        state.update(status="blocked", recovery="settle_only", stop_reason="original operation or limited balance unresolved")
    native = state["episodes"]
    value = {"protocol": PROTOCOL, "operation": "implement", "run_id": context.run_id,
        "request_id": context.request["request_id"], "request_sha256": context.request_sha256,
        **{k: state[k] for k in ("status", "recovery", "stop_reason", "feedback", "assessments")},
        "usage": usage, "code": {"latest": state["latest"], "qualified": state["qualified"]},
        "extensions": {"information": {"backend": "aka.fixed.v1", "fixed_schema": context.request["schema"],
            "native_incumbent": native_incumbent, "native_workspace": state["binding"]["workspace"],
            "coder_episodes": len(native), "promotions": sum(bool(e.get("attempt", {}).get("accepted")) for e in native.values()),
            "diagnostics": state["diagnostics"]}}}
    if state["phase"] == "done" and not usage["unsettled"]:
        state["result"] = value
        state["usage_revision"] = revision
    tx.save("result_ready")
    return value
