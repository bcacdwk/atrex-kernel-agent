"""Schema completion guidance and feedback; synthetic CPU services only."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from aka.contracts.workspace import RevisionContent
from aka.task.candidate_workspace.provider import GitCandidateWorkspace
from aka.workflow.schema.run_state import RunLimits, RunState
from aka.workflow.schema.workflow import SchemaWorkflow, build_feedback
from test_schema_progress import run
from test_schema_qualification_protocol import retained, review, review_harness
from test_schema_workflow import controller, native


def test_actual_episode_prompt_keeps_native_contract_and_schema_completion(tmp_path):
    campaign = native(tmp_path)
    driver, _ = controller(tmp_path, campaign, limits=RunLimits(max_episodes=1))
    assert driver.run(campaign) == "budget: episodes"
    prompt = campaign.agent.calls[0]["prompt"]
    # Inspect the actual combined prompt delivered to the native Coder session.
    assert "Do not claim a speedup merely to terminate; a well-supported pivot is valid." in prompt
    assert "You need not run ABBA just to finish. Report acceptance is not Kernel promotion." in prompt
    assert "Submit `status`, `summary`, and" in prompt
    assert "passing standard full Eval bound to that exact current source" in prompt
    assert "retain that source and submit the normal `candidate_ready` report" in prompt
    assert "even if it is slower than the champion or the performance comparison is unknown" in prompt
    assert "Being slower alone is not a reason to restore the champion or skip qualification." in prompt
    assert "Independent qualification judges Schema fidelity; a passing Eval alone does not establish fidelity." in prompt
    assert "The subsequent native ABBA and policy gates decide global champion promotion." in prompt
    assert "Do not self-qualify the candidate or claim promotion from report acceptance." in prompt
    assert "implementation is incomplete, the design conflicts with the Task or target, bounded attempts are exhausted, or infrastructure prevents completion" in prompt
    assert "An unsuccessful attempt does not require a `candidate_ready` report." in prompt
    assert "existing `summary`, include a `schema_implementation` section" in prompt
    assert "deviations, unresolved assumptions, limitations, and original experiment records" in prompt


def feedback(qualification, *, prepared="task", design="design", public_task="task", hardware="L20D",
             sealed_source_identity=None):
    seed = {"identity": "seed"}
    driver = SchemaWorkflow.__new__(SchemaWorkflow)
    hypothesis = driver._new_hypothesis(seed)
    driver.state = SimpleNamespace(data={"prepared": {"identity": prepared, "public_task": public_task,
        "hardware": hardware}, "design": {"identity": design}, "hypothesis": "h",
        "hypotheses": {"h": hypothesis}, "champion": seed, "tested_candidate": None,
        "working_draft": seed, "episodes": {}, "diagnostics": []})
    binding = {"hypothesis": "h", "prepared_id": prepared, "design": {"identity": design}}
    attempt = {"status": "candidate_ready", "accepted": False}
    if qualification is not None:
        attempt["candidate_qualification"] = qualification
    original = deepcopy((attempt, hypothesis))
    assessment = {}
    assert driver._progress(binding, attempt, [], assessment=assessment,
                            sealed_source_identity=sealed_source_identity) == []
    assert (attempt, hypothesis) == original
    assert assessment["status"] == "no-progress"
    driver.state.data["episodes"]["1"] = {"episode": 1, "progress_assessment": assessment}
    projected = json.loads(build_feedback(driver.state))["episodes"][0]["progress_assessment"]
    assert projected == assessment
    return projected["reasons"]


@pytest.mark.parametrize("qualification,reason", [
    (None, "qualification-receipt-missing"),
    ({}, "qualification-unknown"),
    ({"qualified": False}, "qualification-unknown"),
    ({"qualified": False, "verdict": "malformed"}, "qualification-unknown"),
    ([], "qualification-unknown"),
    ({"qualified": False, "verdict": "fail", "binding": []}, "qualification-binding-mismatch"),
])
def test_missing_or_malformed_result_does_not_claim_a_fidelity_review(qualification, reason):
    assert feedback(qualification) == [reason]


def test_complete_full_eval_cannot_supply_missing_qualification(run):
    item = run.measured("unreviewed-full-eval", latency=1)
    assert item[0].attempts[0].assessment.completeness == "complete"
    before = deepcopy(run.hypothesis)
    assessment = {}
    binding = {"hypothesis": "h", "prepared_id": "task", "design": {"identity": "design"}}
    assert run.driver._progress(binding, {"status": "pivot", "accepted": False},
                                [item], assessment=assessment) == []
    assert assessment == {"status": "no-progress", "reasons": ["qualification-receipt-missing"]}
    assert run.hypothesis == before


@pytest.mark.parametrize("verdict,substantive,reason", [
    ("fail", True, "qualification-failed"),
    ("unknown", True, "qualification-unknown"),
    ("pass", False, "no-substantive-change"),
    ("pass", True, "no-fresh-qualified-record"),
])
def test_feedback_uses_real_qualification_receipt(review_harness, verdict, substantive, reason):
    h = review_harness
    def respond(workspace):
        result = review(workspace, verdict=verdict)
        if not substantive:
            path = workspace / "output/decision.json"
            decision = json.loads(path.read_text())
            decision.update(substantive_change=False, change_evidence=[])
            path.write_text(json.dumps(decision))
        return result
    h.actions.append(respond)
    receipt = h.run()
    assert receipt["verdict"] == verdict and not receipt["diagnostics"]
    # The Schema native callback adds this exact sealed identity to the receipt.
    qualification = {**receipt, "source_identity": h.kwargs["source_identity"]}
    context = dict(prepared=h.kwargs["prepared_id"], design=h.kwargs["design_ref"].identity,
        public_task=h.kwargs["evidence"]["public_task"], hardware=h.kwargs["evidence"]["target_hardware"],
        sealed_source_identity=h.kwargs["source_identity"])
    assert feedback(qualification, **context) == [reason]
    # A stale receipt cannot describe the current candidate, even for fail or
    # unknown outcomes that previously shared the generic unqualified reason.
    for field in ("source_identity", "design_identity", "prepared_id",
                  "public_task_sha256", "target_hardware_sha256"):
        stale = deepcopy(qualification)
        stale["binding"][field] = "different"
        assert feedback(stale, **context) == ["qualification-binding-mismatch"]


@pytest.fixture
def settlement(tmp_path):
    owner = GitCandidateWorkspace()
    candidate = RevisionContent({"kernel.py": b"def copy(x):\n    return x + 0\n"})
    other = RevisionContent({"kernel.py": b"def copy(x):\n    return x\n"})
    candidate_id, other_id = owner.source_identity(candidate), owner.source_identity(other)
    assert candidate_id != other_id

    def settle(qualification, *, sealed=True, prepared="task", design="design",
               public_task="task", hardware="L20D"):
        state = RunState(tmp_path / "settlement", limits=RunLimits(continuous=True),
            workspace=tmp_path / "settlement-native", identity={"synthetic": True})
        driver = SchemaWorkflow.__new__(SchemaWorkflow)
        driver.state = state
        campaign = SimpleNamespace(candidate_workspace=owner, workspace=tmp_path / "settlement-native")
        with state.owned():
            seed = state.retain_source(other, owner=owner, origin={"role": "champion", "verified": True})
            tested = state.retain_source(other, owner=owner, origin={"role": "tested_candidate"})
            previous = {"episode": 1, "progress": [], "progress_assessment": {
                "status": "no-progress", "reasons": ["qualification-failed"]}}
            state.data.update(champion=seed, tested_candidate=tested, working_draft=seed,
                prepared={"identity": prepared, "public_task": public_task, "hardware": hardware},
                design={"identity": design}, hypothesis="h", episodes={"1": deepcopy(previous)},
                window={"episodes": [1], "progress": []}, seen_records=["previous-record"])
            state.data["hypotheses"]["h"] = driver._new_hypothesis(seed)
            state.data["counts"].update(windows=1, episodes=2, settled_episodes=1, tokens=17)
            before_counts = deepcopy(state.data["counts"])
            before_hypothesis = deepcopy(state.data["hypotheses"]["h"])
            binding = {"episode": 2, "prepared_id": prepared, "design": {"identity": design}, "hypothesis": "h"}
            attempt = {"episode": 2, "status": "candidate_ready", "accepted": False, "tokens": 17}
            if qualification is not None:
                attempt["candidate_qualification"] = qualification
            original_attempt = deepcopy(attempt)
            facts = {"active": None, "attempt": attempt, "records": (), "draft": other,
                     "champion": other, "candidate": candidate if sealed else None}
            driver._settle(campaign, binding, facts)
            row = state.data["episodes"]["2"]
            if sealed:
                assert row["sealed_candidate"]["identity"] == candidate_id
            else:
                assert row["sealed_candidate"] is None
            assert row["progress"] == [] and row["progress_assessment"]["status"] == "no-progress"
            assert state.data["working_draft"]["identity"] == other_id
            assert state.data["champion"]["identity"] == other_id
            assert state.data["tested_candidate"] == tested
            assert state.data["hypotheses"]["h"] == before_hypothesis
            assert state.data["episodes"]["1"] == previous
            assert state.data["seen_records"] == ["previous-record"]
            assert state.data["counts"] == {**before_counts, "settled_episodes": 2}
            assert state.data["window"] == {"episodes": [1, 2], "progress": []}
            assert attempt == original_attempt and row["tokens"] == 17
            projected = json.loads(build_feedback(state))["episodes"][-1]["progress_assessment"]
            assert projected == row["progress_assessment"]
            saved = state.path.read_bytes()
            driver._settle(campaign, binding, facts)
            assert state.path.read_bytes() == saved  # Reentry cannot rewrite settled feedback or count twice.
            return projected["reasons"]
    return SimpleNamespace(settle=settle, candidate=candidate, other=other,
                           candidate_id=candidate_id, other_id=other_id)


@pytest.mark.parametrize("verdict", ["fail", "unknown"])
@pytest.mark.parametrize("source", ["candidate", "other"])
def test_settled_feedback_binds_negative_review_to_actual_sealed_source(review_harness, settlement, verdict, source):
    h = review_harness
    h.kwargs.update(source=getattr(settlement, source), source_identity=getattr(settlement, source + "_id"))
    h.actions.append(lambda workspace: review(workspace, verdict=verdict))
    receipt = h.run()
    assert receipt["verdict"] == verdict and receipt["qualified"] is False and not receipt["diagnostics"]
    qualification = {**receipt, "source_identity": h.kwargs["source_identity"]}
    assert qualification["binding"]["source_identity"] == qualification["source_identity"]
    original = retained(h.directory)
    reason = settlement.settle(qualification, prepared=h.kwargs["prepared_id"],
        design=h.kwargs["design_ref"].identity, public_task=h.kwargs["evidence"]["public_task"],
        hardware=h.kwargs["evidence"]["target_hardware"])
    expected = ("qualification-failed" if verdict == "fail" else "qualification-unknown") \
        if source == "candidate" else "qualification-binding-mismatch"
    assert reason == [expected]
    assert retained(h.directory) == original


@pytest.mark.parametrize("qualification,reason", [
    (None, "qualification-receipt-missing"),
    ({}, "qualification-unknown"),
    ([], "qualification-unknown"),
    ({"qualified": False, "verdict": "fail"}, "qualification-unknown"),
    ({"qualified": False, "verdict": "fail", "binding": []}, "qualification-binding-mismatch"),
])
def test_settlement_without_bound_review_does_not_claim_failure(settlement, qualification, reason):
    assert settlement.settle(qualification) == [reason]


def test_fail_without_actual_sealed_source_stays_unknown(settlement):
    # The result matches seed, working draft and tested source; none proves a seal.
    qualification = {"qualified": False, "verdict": "fail", "source_identity": settlement.other_id}
    assert settlement.settle(qualification, sealed=False) == ["qualification-unknown"]


def test_settlement_after_lost_review_receipt_stays_unknown(review_harness, settlement):
    h = review_harness
    h.kwargs.update(source=settlement.candidate, source_identity=settlement.candidate_id)
    def lost(workspace):
        raise RuntimeError("synthetic missing original process receipt")
    h.actions.extend([lambda workspace: review(workspace, nested=True), lost])
    with pytest.raises(RuntimeError, match="synthetic missing original process receipt"):
        h.run()
    receipt = h.run()
    assert receipt["verdict"] == "unknown" and receipt["qualified"] is False
    assert receipt["diagnostics"][-1]["category"] == "invocation.unknown"
    assert len(h.calls) == h.state.data["counts"]["agent_calls"] == 2
    assert not (h.directory / "result.json").exists()
    original = retained(h.directory)
    assert settlement.settle({**receipt, "source_identity": settlement.candidate_id},
        prepared=h.kwargs["prepared_id"], design=h.kwargs["design_ref"].identity,
        public_task=h.kwargs["evidence"]["public_task"],
        hardware=h.kwargs["evidence"]["target_hardware"]) == ["qualification-unknown"]
    assert retained(h.directory) == original
