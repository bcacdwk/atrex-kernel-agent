"""Synthetic stage interruptions use real durable runs and one native history."""
import json

import pytest

from kernelschema.artifacts import Bundle
from kernelschema.contracts import ProtocolError
from aka.workflow.schema import backend
from aka.workflow.schema.dynamic import DynamicWorkflow
from aka.workflow.light.long_horizon.store import CampaignStore
from test_schema_dynamic import dynamic, fixed, run, close


def reopen(d):
    d.workflow.close()
    d.workflow = backend.create_workflow(d.selection, d.modules)


def planner_calls(d):
    return [json.loads(row) for fake in d.fake_dirs if (fake / "invocations.jsonl").exists()
            for row in (fake / "invocations.jsonl").read_text().splitlines()
            if json.loads(row)["role"] == "planner"]


@pytest.mark.parametrize("boundary", ["writing_published", "new_request_ready", "new_request_bound", "new_e_unadmitted"])
def test_revise_published_and_admission_interruptions(dynamic, boundary):
    d = dynamic(decision="revise", action="implement")
    fired = []
    def checkpoint(label, workflow):
        if label == boundary and workflow.state["phase"] == "followup" and not fired:
            fired.append(label)
            raise KeyboardInterrupt("synthetic stage interruption")
    d.workflow.checkpoint = checkpoint
    try:
        with pytest.raises(KeyboardInterrupt):
            run(d)
        summary = d.workflow.summary()
        assert len(summary["history"]) == 2
        old = Bundle(d.workflow.root).document(summary["history"][0]["result"])
        assert old["code"]["qualified"] is not None
        assert summary["current_qualified"] is None
        assert summary["status"] == "partial"
        assert summary["native_result"]["schema"] != summary["schema"]
        schema = summary["schema"]
        operations = dict(d.owner.operations())
        assert len(planner_calls(d)) == 1
        reopen(d)
        result = run(d)
        assert result["status"] == "completed" and result["schema"] == schema
        assert len(planner_calls(d)) == 1 and len(result["history"]) == 3
        assert all(d.owner.operations()[key] == value for key, value in operations.items())
        assert CampaignStore.checkpoint(d.request.workspace)[0].episodes == 2
    finally:
        close(d)


def test_lost_fixed_result_recovers_before_planner(dynamic):
    d = dynamic()
    fired = []
    def checkpoint(label, workflow):
        if label == "child_result_before_driver" and workflow.state["phase"] == "fixed" and not fired:
            fired.append(label)
            # Original run has durable native receipts; outer result is lost.
            (workflow.root / workflow.state["current"]["out"] / "result.json").unlink()
            raise KeyboardInterrupt("synthetic caller result loss")
    d.workflow.checkpoint = checkpoint
    try:
        with pytest.raises(KeyboardInterrupt):
            run(d)
        assert len(d.trace) == 2 and planner_calls(d) == []
        operations = dict(d.owner.operations())
        reopen(d)
        result = run(d)
        assert result["status"] == "completed" and len(d.trace) == 2
        assert all(d.owner.operations()[key] == value for key, value in operations.items())
        assert len(planner_calls(d)) == 1
    finally:
        close(d)


def test_finishing_is_resumed_before_new_stage(dynamic):
    d = dynamic()
    fired = []
    def tx_checkpoint(label):
        if label == "finishing" and not fired:
            fired.append(label)
            raise KeyboardInterrupt("synthetic finish interruption")
    def checkpoint(label, workflow):
        if label == "new_e_unadmitted":
            workflow.controller.tx.checkpoint = tx_checkpoint
    d.workflow.checkpoint = checkpoint
    try:
        with pytest.raises(KeyboardInterrupt):
            run(d)
        assert d.workflow.controller.tx.state["phase"] == "finishing"
        assert planner_calls(d) == []
        reopen(d)
        result = run(d)
        assert result["status"] == "completed" and len(d.trace) == 2
        first = result["history"][0]
        from kernelschema.api import open_run
        state = open_run("implement", d.workflow.root / first["out"]).load_state("aka-fixed-backend")
        assert state["phase"] == "done" and state["finished"]
        assert len(planner_calls(d)) == 1
    finally:
        close(d)


def test_bootstrap_missing_binding_never_allocates_or_plans(dynamic):
    d = dynamic()
    def checkpoint(label, workflow):
        if label == "new_request_bound":
            out = workflow.root / workflow.state["current"]["out"]
            (out / ".runtime/state/budget.json").unlink()
            raise KeyboardInterrupt("synthetic lost bootstrap binding")
    d.workflow.checkpoint = checkpoint
    try:
        with pytest.raises(KeyboardInterrupt):
            run(d)
        before = d.owner.operations()
        reopen(d)
        with pytest.raises(ProtocolError, match="owner binding missing"):
            run(d)
        assert d.owner.operations() == before == {}
        assert planner_calls(d) == [] and d.trace == []
    finally:
        close(d)


def test_same_r_multiple_episodes_do_not_trigger_planner(dynamic):
    d = dynamic(reviews=("fail", "pass"))
    try:
        result = run(d)
        assert result["status"] == "completed", result
        assert len([r for r in d.trace if r["role"] == "coder"]) == 2
        assert len(planner_calls(d)) == 1
        first = Bundle(d.workflow.root).document(result["history"][0]["result"])
        assert first["extensions"]["information"]["coder_episodes"] == 2
    finally:
        close(d)


@pytest.mark.parametrize("reviews", [("fail",), ("unknown",)])
def test_ordinary_partial_or_unknown_never_plans(dynamic, reviews):
    d = dynamic(reviews=reviews)
    try:
        result = run(d)
        assert result["status"] != "completed" and result["optimize_count"] == 0
        assert len(result["history"]) == 1 and planner_calls(d) == []
    finally:
        close(d)


def test_paused_parent_does_not_optimize(dynamic):
    d = dynamic()
    d.owner.pause()
    try:
        result = run(d)
        assert result["status"] != "completed" and result["optimize_count"] == 0
        assert planner_calls(d) == [] and d.trace == []
    finally:
        close(d)


def test_unknown_native_owner_fences_planner_and_new_r(dynamic):
    d = dynamic(lose_native=True)
    try:
        with pytest.raises(RuntimeError, match="unknown original"):
            run(d)
        current = dict(d.workflow.state["current"])
        operations = dict(d.owner.operations())
        assert d.owner.fences() and planner_calls(d) == []
        reopen(d)
        result = run(d)
        assert result["status"] == "blocked" and result["recovery"] == "settle_only"
        assert d.workflow.state["current"] == current
        current_ops = d.owner.operations()
        assert current_ops.keys() == operations.keys() and planner_calls(d) == []
        for key, original in operations.items():
            for field in ("reservation", "history", "latest", "launched", "terminated"):
                assert current_ops[key][field] == original[field]
    finally:
        close(d)


def test_exhausted_planner_budget_does_not_launch(dynamic):
    d = dynamic(planner_limit=0)
    try:
        result = run(d)
        assert result["status"] != "completed" and result["optimize_count"] == 0
        assert len(d.trace) == 2 and planner_calls(d) == []
    finally:
        close(d)


def test_paid_production_finish_receipt_precedes_planner_on_resume(dynamic):
    d = dynamic(production=True)
    retained = []
    def tx_checkpoint(label):
        tx = d.workflow.controller.tx
        if label == "native_invocation_retained" and tx.state["phase"] == "finishing" and not retained:
            op = next(op for op in tx.state["operations"].values()
                      if op["role"] == "reviewer" and not op["terminal"] and tx.invocation(op) is not None)
            retained.append(dict(op))
            raise KeyboardInterrupt("synthetic paid production finish return loss")
    def checkpoint(label, workflow):
        if label == "new_e_unadmitted":
            workflow.controller.tx.checkpoint = tx_checkpoint
    d.workflow.checkpoint = checkpoint
    try:
        with pytest.raises(KeyboardInterrupt):
            run(d)
        assert retained and d.workflow.controller.tx.state["phase"] == "finishing"
        assert planner_calls(d) == []
        original = retained[0]
        reopen(d)
        result = run(d)
        assert result["status"] == "completed", result
        assert len(planner_calls(d)) == 1
        assert sum(row["session"] == original["session_id"] for row in d.trace) == 1
        settled = d.owner.operations()[original["id"]]
        assert settled["reservation"] == original["reservation"]
        assert settled["terminated"] and len(settled["history"]) == 1
    finally:
        close(d)
