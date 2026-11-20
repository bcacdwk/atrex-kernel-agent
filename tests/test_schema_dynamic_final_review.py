"""Independent CPU review of the finite AKA chain's actual admission paths."""
import json
import pytest

from kernelschema.contracts import UnsupportedError
from kernelschema.artifacts import Bundle
from test_schema_dynamic import dynamic, fixed, run, close
from test_schema_dynamic_recovery import planner_calls


@pytest.mark.parametrize("boundary,metric", [
    ("request", "coder"), ("request", "evaluator_jobs"),
    ("request", "active_ms"), ("native", "model_calls"),
])
def test_design_cannot_spend_with_an_impossible_fixed_finish(dynamic, boundary, metric):
    limits = {"calls": {"planner": 0, "reviewer": 8, "coder": 4, "extract": 0},
              "evaluator_jobs": 8, "active_ms": 120000, "tokens": None}
    if metric == "coder":
        limits["calls"]["coder"] = 0
    else:
        limits[metric] = {"evaluator_jobs": 1, "active_ms": 3999, "model_calls": 1}[metric]
    d = dynamic(start="design", optimize=False,
                **{"fixed_limits" if boundary == "request" else "fixed_native_limits": limits})
    try:
        with pytest.raises(UnsupportedError, match="next fixed.*minimum finish"):
            run(d)
        assert planner_calls(d) == [] and d.trace == []
        assert d.owner.operations() == {}
    finally:
        close(d)


def test_explicit_source_implement_action_reaches_the_next_native_coder(dynamic):
    d = dynamic(decision="keep", action="implement")
    scenario = d.fake_dirs[0] / "scenario.json"
    value = json.loads(scenario.read_text())
    value["draft"]["action_inputs"] = ["evidence:0"]
    scenario.write_text(json.dumps(value))
    try:
        result = run(d)
        assert result["status"] == "completed" and len(result["history"]) == 3
        old = Bundle(d.workflow.root).document(result["history"][0]["result"])
        followup = Bundle(d.workflow.root).document(result["history"][-1]["request"])
        assert followup["mode"] == "code" and followup["seed"] == old["code"]["latest"]
        assert [row for row in d.trace if row["role"] == "coder"][-1]["source"] == followup["seed"]
        assert result["history"][-1]["native"]["episodes"] == 2
    finally:
        close(d)


def test_bound_revised_request_still_obeys_the_original_parent_pause(dynamic):
    d = dynamic(decision="revise", action="implement")
    original = {}
    def checkpoint(label, workflow):
        if label == "new_request_bound" and workflow.state["phase"] == "followup":
            original.update(d.owner.operations())
            d.owner.pause()
    d.workflow.checkpoint = checkpoint
    try:
        result = run(d)
        assert result["status"] == "partial" and result["current_qualified"] is None
        assert d.owner.operations() == original
        assert len([row for row in d.trace if row["role"] == "coder"]) == 1
        old = Bundle(d.workflow.root).document(result["history"][0]["result"])
        assert old["code"]["qualified"] is not None
        assert result["history"][0]["schema"] != result["schema"]
        assert result["native_result"]["schema"] == result["schema"]
        assert result["native_result"]["incumbent"] is not None
    finally:
        close(d)


def test_no_schema_start_executes_the_complete_finite_chain(dynamic):
    d = dynamic(start="design", decision="keep", action="implement")
    try:
        result = run(d)
        assert result["status"] == "completed" and result["optimize_count"] == 1
        requests = [Bundle(d.workflow.root).document(row["request"]) for row in result["history"]]
        assert [q["operation"] for q in requests] == ["design", "implement", "optimize", "implement"]
        assert "schema" not in requests[0]
        assert len({q["budget"]["parent"]["sha256"] for q in requests}) == 1
        assert len(planner_calls(d)) == 2
        assert len([row for row in d.trace if row["role"] == "coder"]) == 2
        assert result["history"][-1]["native"]["episodes"] == 2
        assert result["current_qualified"] is not None
        assert sum(op["latest"]["evaluator_jobs"] for op in d.owner.operations().values()) == 3
    finally:
        close(d)
