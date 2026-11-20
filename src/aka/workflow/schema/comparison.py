"""The real native ABBA callback, with an explicit same-allocation service.

An independent full-Task timing is never relabeled as native comparison. The
selected service must execute the original schedule in one allocation and
return each unchanged Task report plus the allocation's cumulative receipt.
"""
from kernelschema.artifacts import Publication
from kernelschema.contracts import ProtocolError, digest
from aka.workflow.light.long_horizon.models import VerificationResult
from aka.workflow.light.long_horizon.verifier import verification_schedule, score_verification_payload

from .calls import Unsettled
from .source import publish_native_source


class ComparisonCallback:
    def __init__(self, controller, channel):
        self.controller, self.channel = controller, channel

    def __call__(self, workspace, argv):
        # The real RecordedABBAValidator dispatches its explicit trusted callback
        # before constructing an Atrex-Bench request. Ad-hoc calls stay forbidden.
        raise ProtocolError("Schema comparison requires the native sealed revision binding")

    def schema_comparison(self, verifier, workspace, *, base_commit, candidate_commit, changed_paths):
        c, tx = self.controller, self.controller.tx
        if changed_paths != ["kernel.py"]:
            return VerificationResult("ERROR", None, None, None,
                                      error="native single-file acceptance requires a kernel-only candidate")
        episode = c.active_episode()
        audit = tx.state["episodes"].get(str(episode), {}).get("qualification")
        if not audit or audit["qualified"] is not True or audit["candidate_commit"] != candidate_commit:
            raise ProtocolError("ABBA requires durable qualification for the exact sealed candidate")
        if self.channel is None:
            return VerificationResult("ERROR", None, None, None, error="no qualified same-allocation full-Task comparison service")
        admission = self.channel.admission(c.run)
        if admission.get("status") != "pass":
            return VerificationResult("ERROR", None, None, None, error=admission["reason"])
        incumbent = publish_native_source(c.run.root, workspace, c.campaign.candidate_workspace, revision=base_commit)
        candidate = publish_native_source(c.run.root, workspace, c.campaign.candidate_workspace,
                                          revision=candidate_commit, expected=audit["source"])
        source = candidate.source
        key = f"e{episode}-abba"
        op = tx.plan(key, role="evaluator", source=source, episode=episode, purpose="native same-allocation full-Task ABBA")
        tx.reserve(op, floor=tx.finish_floor(evaluate=False,
            review=2 if c.campaign.optimization_mode == "production" else 0))
        schedule = verification_schedule(verifier.repeats)
        spec = {"operation_id": op["id"], "submission_key": op["id"], "owner": c.run.owner,
            "reservation": op["reservation"], "task": c.run.request["task"], "scope": c.run.request["scope"],
            "candidate": source, "incumbent": incumbent.source, "base_commit": base_commit,
            "candidate_commit": candidate_commit, "schedule": schedule, "per_run_timeout": verifier.per_run_timeout,
            "qualification_sha256": digest(audit)}
        if op.get("spec") is None:
            op["spec"] = spec
            tx.intent(op, self.channel.locator(op["id"]))
            self.channel.prepare(c.run, spec)
            tx.save("comparison_prepared")
        elif op["spec"] != spec:
            raise ProtocolError("replayed ABBA differs from the original fixed comparison")
        self.channel.validate(c.run, spec)
        owned = tx.owner.operations(c.run.run_id)[op["id"]]
        if not owned["launched"]:
            self.channel.submit(c.run, spec, authorize=lambda: tx.owner.authorize(op["reservation"]))
        terminal = self.channel.settle(c.run, spec, Publication(c.run.root, op["publication_id"]))
        if terminal.get("state") != "terminal":
            tx.unknown(op, "original same-allocation comparison remains unresolved")
            raise Unsettled(tx.state["stop_reason"])
        tx.absorb(op, terminal["usage_row"], evidence=terminal.get("evidence", ()))
        receipt = terminal["comparison"]
        if (receipt.get("schedule") != schedule or not receipt.get("allocation_id")
                or receipt.get("same_allocation") is not True or receipt.get("measurement_scope") != "full-task"
                or receipt.get("operation_id") != op["id"] or receipt.get("spec_sha256") != digest(spec)
                or len(receipt.get("runs", [])) != len(schedule)):
            raise ProtocolError("comparison lacks original allocation/order/Task/source evidence")
        rows = []
        for expected, row in zip(schedule, receipt["runs"], strict=True):
            ref = source if expected["revision"] == "candidate" else incumbent.source
            if (row.get("revision") != expected["revision"] or row.get("repeat") != expected["repeat"]
                    or row.get("allocation_id") != receipt["allocation_id"]):
                raise ProtocolError("comparison row escaped original ABBA allocation/order")
            report = c.run.bundle.document(row["report"], "evaluation")
            if any(report["subject"].get(k) != v for k, v in {
                    "task_sha256": c.run.request["task"]["sha256"], "source_sha256": ref["sha256"],
                    "evaluator_sha256": c.run.objects["task"]["evaluator"]["sha256"]}.items()):
                raise ProtocolError("comparison full Task report belongs to another subject")
            # Service checks original loaded bytes/input/environment/receipt;
            # the callback retains its complete unchanged report and samples.
            measured = self.channel.measurement(c.run, spec, row)
            if measured["measurement_scope"] != "full-task" or set(measured["latency_us_by_shape"]) != set(c.coverage_ids):
                raise ProtocolError("comparison measurement does not cover the full Task")
            passing = all(report[k]["status"] == "pass" for k in ("build", "correctness", "performance"))
            rows.append({**expected, "exit_code": 0 if passing else 1,
                         "result": {**measured, "all_pass": passing}})
        scored = score_verification_payload({"schema_version": 1, "runs": rows}, schedule=schedule,
            repeats=verifier.repeats, min_improvement_pct=verifier.min_improvement_pct,
            artifact=terminal["usage_row"]["receipt_id"])
        op["comparison"] = receipt
        tx.save("comparison_settled")
        return scored
