"""CPU-only native receipt -> settlement -> new D/E request feedback regressions."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from aka.contracts.workspace import CandidateHandle
from aka.workflow.light.long_horizon.campaign import LongHorizonCampaign
from aka.workflow.light.long_horizon.models import EpisodeHandoff
from aka.workflow.light.long_horizon.session import LongSessionRunner
from aka.workflow.light.long_horizon.store import CampaignStore
from aka.workflow.schema.materials import DesignRef, read_design
from aka.workflow.schema.qualification_feedback import MAX_CHARACTERS, project
from aka.workflow.schema.run_state import RunLimits, digest
from aka.workflow.schema.workflow import build_feedback
from test_schema_workflow import controller, native


REASON = "Synthetic fidelity failure: design requires four slots; sealed candidate uses three."
OBLIGATION = "root.DESIGN.Tile_pipeline"
sha = lambda raw: hashlib.sha256(raw.encode() if isinstance(raw, str) else raw).hexdigest()


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    def run(*, legacy=False, verdict="fail", pause=False, episodes=2):
        campaign = native(tmp_path)
        sealed, receipts, snapshots, old_rows = {}, [], [], []
        def qualify(*, binding, source, source_identity, **kwargs):
            design = read_design(DesignRef.from_dict(binding["design"]))
            receipt = {"qualified": False, "verdict": verdict, "substantive_change": True,
                "source_identity": source_identity, "operation_id": binding["operation_id"] + ":qualification",
                "binding": {"prepared_id": binding["prepared_id"], "design_identity": binding["design"]["identity"],
                    "source_identity": source_identity,
                    "public_task_sha256": sha(driver.state.data["prepared"]["public_task"]),
                    "target_hardware_sha256": sha(driver.state.data["prepared"]["hardware"]),
                    "sealed_files": {name: sha(raw) for name, raw in source.files.items()}},
                "evidence": [{"id": OBLIGATION, "verdict": verdict, "reason": REASON,
                    "evidence": [{"source": alias, "start": 1, "end": 1, "sha256": sha(raw)}
                        for alias, raw in (("candidate/kernel.py", source.files["kernel.py"]),
                            ("current-design/root.yaml", design.checked.files["root.yaml"]),
                            ("task/hardware.md", b"L20D"))]}],
                "directory": "/private/qualification", "private_cause": "PRIVATE_SENTINEL",
                "thinking": "THINKING_SENTINEL", "diagnostics": [{"exception": "EXCEPTION_SENTINEL"}]}
            receipts.append(deepcopy(receipt))
            return receipt
        driver, planner = controller(tmp_path, campaign, limits=RunLimits(max_episodes=episodes),
            candidate_qualification=qualify)
        original_run, original_facts = LongSessionRunner.run, campaign.episode_facts
        def ready(self, workspace, *args, **kwargs):
            result = original_run(self, workspace, *args, **kwargs)
            _, active = CampaignStore.checkpoint(campaign.workspace)
            if active["episode"] != 1:
                return result
            handle = CandidateHandle(active["episode"], active["base_commit"], active["episode_branch"],
                type(campaign.workspace)(active["worktree"]))
            source = campaign.candidate_workspace.read_source(workspace)
            commit = campaign.candidate_workspace.seal_candidate(handle, source)
            sealed[1] = source
            return replace(result, exit_status=0, timed_out=False,
                handoff=EpisodeHandoff("candidate_ready", candidate_commit=commit), completion_diagnosis="")
        def facts(episode):
            result = original_facts(episode)
            if episode in sealed:
                result["candidate"] = sealed[episode]
            return result
        monkeypatch.setattr(LongSessionRunner, "run", ready)
        monkeypatch.setattr(campaign, "episode_facts", facts)
        monkeypatch.setattr(LongHorizonCampaign, "_completion_check", lambda *args: "")
        monkeypatch.setattr("aka.workflow.light.long_horizon.main_adapter.candidate_policy_violations",
            lambda *args, **kwargs: [])
        original_settle = driver._settle
        def settle(*args, **kwargs):
            original_settle(*args, **kwargs)
            if len(driver.state.data["episodes"]) == 1:
                if legacy:
                    driver.state.data["episodes"]["1"].pop("qualification_feedback", None)
                    driver.state.save()
                old_rows.append(deepcopy(driver.state.data["episodes"]["1"]))
                snapshots.append(json.loads(build_feedback(driver.state,
                    qualification_feedback=driver._qualification_feedback(campaign))))
        monkeypatch.setattr(driver, "_settle", settle)
        original_engine = campaign.create_engine
        class Pause(Exception):
            pass
        def create_engine(**kwargs):
            if pause and driver.state.data["binding"]["episode"] == 2:
                raise Pause()
            return original_engine(**kwargs)
        monkeypatch.setattr(campaign, "create_engine", create_engine)
        if pause:
            with pytest.raises(Pause):
                driver.run(campaign)
        else:
            assert driver.run(campaign) == "budget: episodes"
        return SimpleNamespace(driver=driver, campaign=campaign, planner=planner, receipts=receipts,
            snapshots=snapshots, old_rows=old_rows, sealed=sealed, original_engine=original_engine)
    return run


@pytest.mark.parametrize("legacy", [False, True])
def test_native_archive_settlement_next_prompt_and_planner(delivery, legacy):
    h = delivery(legacy=legacy)
    data = h.driver.state.data
    attempt = CampaignStore.checkpoint(h.campaign.workspace)[0].attempts[0]
    first, following = data["episodes"]["1"], data["episodes"]["2"]["binding"]
    assert attempt["status"] == "candidate_ready" and not attempt["accepted"]
    assert attempt["candidate_qualification"] == h.receipts[0]
    assert first == h.old_rows[0]  # Legacy recovery is a view, never backfill.
    assert first["progress_assessment"]["reasons"] == ["qualification-failed"]
    assert following["seed"]["identity"] == first["sealed_candidate"]["identity"]
    prompt = h.campaign.agent.calls[1]["prompt"]
    assert following["context"] in prompt and following["context_id"] == digest(following["context"])
    assert REASON in prompt and OBLIGATION in prompt
    feedback = h.snapshots[0]["qualification_feedback"]
    assert feedback["verdict"] == "fail" and feedback["receipt_sha256"] == digest(h.receipts[0])
    assert feedback["operation_id"] == h.receipts[0]["operation_id"]
    assert feedback["reviewed"]["source_identity"] == following["seed"]["identity"]
    citations = feedback["obligations"][0]["evidence"]
    assert {item["source"] for item in citations} == {
        "candidate/kernel.py", "current-design/root.yaml", "task/hardware.md"}
    for citation in citations:
        assert citation["sha256"] in prompt and citation["excerpt"] in prompt
    assert citations[-1]["excerpt"] == "1: L20D"
    for private in ("/private/qualification", "PRIVATE_SENTINEL", "THINKING_SENTINEL", "EXCEPTION_SENTINEL"):
        assert private not in prompt and private not in json.dumps(h.snapshots[0])
    assert len(h.campaign.agent.calls) == 2 and len(h.receipts) == 1
    assert data["counts"]["tokens"] == 14 and data["counts"]["settled_episodes"] == 2
    assert len(build_feedback(h.driver.state)) <= 8000


def test_new_view_does_not_change_frozen_binding_on_replay(delivery, monkeypatch):
    h = delivery(legacy=True, pause=True)
    frozen = deepcopy(h.driver.state.data["binding"])
    native_before = (h.campaign.workspace / ".atrex_long_horizon/episodes/e0001/attempt.json").read_bytes()
    monkeypatch.setattr(h.driver, "_qualification_feedback", lambda *args, **kwargs:
        pytest.fail("frozen binding must not be rebuilt"))
    monkeypatch.setattr(h.campaign, "create_engine", h.original_engine)
    assert h.driver.run(h.campaign) == "budget: episodes"
    assert h.driver.state.data["episodes"]["2"]["binding"] == frozen
    assert h.driver.state.data["episodes"]["1"] == h.old_rows[0]
    assert (h.campaign.workspace / ".atrex_long_horizon/episodes/e0001/attempt.json").read_bytes() == native_before
    assert len(h.campaign.agent.calls) == 2 and h.driver.state.data["counts"]["tokens"] == 14


def test_old_receipt_reaches_real_new_optimize_request(delivery):
    h = delivery(legacy=True, episodes=5)
    request = h.planner.calls[1][1]
    feedback = json.loads(request.feedback[0]["text"])
    summary = feedback["qualification_feedback"]
    assert summary["verdict"] == "fail" and summary["original_episode"] == 1
    assert summary["scope"] == "historical-reviewed-source"
    assert summary["obligations"][0]["reason"] == REASON
    assert summary["obligations"][0]["evidence"][0]["excerpt"]
    assert len(request.feedback[0]["text"]) <= 8000
    assert len(h.campaign.agent.calls) == 5 and len(h.receipts) == 1


def test_bound_unknown_is_delivered_without_becoming_fail_or_legal(delivery):
    h = delivery(verdict="unknown")
    summary = h.snapshots[0]["qualification_feedback"]
    assert summary["status"] == "reviewed" and summary["verdict"] == "unknown"
    assert summary["obligations"][0]["verdict"] == "unknown"
    assert REASON in h.campaign.agent.calls[1]["prompt"]
    assert h.driver.state.data["episodes"]["1"]["progress_assessment"]["reasons"] == ["qualification-unknown"]
    assert not h.driver.state.data["episodes"]["1"]["accepted"]


@pytest.fixture
def projection():
    files = {"kernel.py": b"NUM_STAGES = 3\nload()\n"}
    prepared = {"identity": "prepared", "public_task": "Public CPU Task", "hardware": "L20D"}
    binding = {"episode": 2, "operation_id": "a00000008", "prepared_id": "prepared",
        "design": {"identity": "design"}}
    receipt = {"verdict": "fail", "source_identity": "sealed", "operation_id": "a00000008:qualification",
        "binding": {"prepared_id": "prepared", "design_identity": "design", "source_identity": "sealed",
            "public_task_sha256": sha(prepared["public_task"]), "target_hardware_sha256": sha("L20D"),
            "sealed_files": {"kernel.py": sha(files["kernel.py"])}},
        "evidence": [{"id": OBLIGATION, "verdict": "fail", "reason": REASON,
            "evidence": [{"source": "candidate/kernel.py", "start": 1, "end": 2,
                "sha256": sha(files["kernel.py"])}]}]}
    kwargs = dict(binding=binding, prepared=prepared, sealed_identity="sealed", source_files=files,
        design_files={"root.yaml": b"public design\n"}, obligations={OBLIGATION})
    return receipt, kwargs


@pytest.mark.parametrize("field", ["prepared_id", "design_identity", "source_identity",
    "public_task_sha256", "target_hardware_sha256", "sealed_files", "operation_id", "stale"])
def test_wrong_binding_never_becomes_seed_repair_guidance(projection, field):
    receipt, kwargs = projection
    if field == "operation_id":
        receipt[field] = "different:qualification"
    elif field == "stale":
        receipt["source_identity"] = receipt["binding"]["source_identity"] = "stale"
    else:
        receipt["binding"][field] = "different"
    result = project(receipt, **kwargs)
    assert result["status"] == "unavailable" and result["verdict"] == "unknown"
    assert REASON not in json.dumps(result)


@pytest.mark.parametrize("receipt", [None, [], {}, {"verdict": "unknown"}])
def test_missing_or_unknown_receipt_is_honest(projection, receipt):
    _, kwargs = projection
    result = project(receipt, **kwargs)
    assert result["status"] == "unavailable" and result["verdict"] == "unknown"


@pytest.mark.parametrize("change", ["source", "hash", "range", "seed", "bool"])
def test_invalid_citation_does_not_disclose_its_path(projection, change):
    receipt, kwargs = projection
    citation = receipt["evidence"][0]["evidence"][0]
    if change == "source":
        citation["source"] = "/private/do-not-project"
    elif change == "hash":
        citation["sha256"] = "0" * 64
    elif change == "range":
        citation["end"] = 300
    elif change == "seed":
        citation["source"] = "initial-seed/kernel.py"
    else:
        citation["start"] = True
    result = project(receipt, **kwargs)
    row = result["obligations"][0]
    assert result["verdict"] == "fail" and row["reason"] == REASON
    assert row["unavailable_citations"] == 1 and row["evidence"] == []
    assert "/private" not in json.dumps(result)


def test_long_public_reason_is_bounded_and_preserves_failed_ids(projection):
    receipt, kwargs = projection
    receipt["evidence"][0]["reason"] = "/private/host-path " + REASON * 3000
    for number in range(5):
        row = deepcopy(receipt["evidence"][0])
        row["id"] = f"child{number}.DESIGN.Tile_pipeline"
        kwargs["obligations"].add(row["id"])
        receipt["evidence"].append(row)
    result = project(receipt, **kwargs)
    text = json.dumps(result, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    assert len(text) <= MAX_CHARACTERS and "/private" not in text
    assert result["verdict"] == "fail" and result["receipt_sha256"] == digest(receipt)
    assert {row["id"] for row in result["obligations"]} == kwargs["obligations"]
    assert all(row["evidence"] and row["evidence"][0]["sha256"] == sha(kwargs["source_files"]["kernel.py"])
        for row in result["obligations"])
    state = SimpleNamespace(data={"prepared": kwargs["prepared"], "design": kwargs["binding"]["design"],
        "hypothesis": "hypothesis", "hypotheses": {"hypothesis": {"milestones": ["detail" * 3000]}},
        "champion": {"identity": "champion", "detail": "detail" * 3000},
        "tested_candidate": None, "working_draft": None, "diagnostics": [],
        "episodes": {str(number): {"episode": number, "binding": kwargs["binding"],
            "status": "candidate_ready", "accepted": False, "records": ["record" * 3000],
            "measurements": [], "qualification_feedback": result} for number in range(1, 5)}})
    feedback = build_feedback(state, qualification_feedback=result)
    assert len(feedback) <= 8000 and OBLIGATION in feedback and result["receipt_sha256"] in feedback


def test_new_request_rejects_other_current_task_design_hardware_or_seed(delivery, monkeypatch):
    h = delivery(legacy=True)
    data = h.driver.state.data
    source = data["episodes"]["1"]["sealed_candidate"]["identity"]
    original = deepcopy(data)
    for changed in ("identity", "public_task", "hardware", "design", "source"):
        h.driver.state.data = deepcopy(original)
        if changed == "design":
            h.driver.state.data["design"]["identity"] = "different"
        elif changed != "source":
            h.driver.state.data["prepared"][changed] = "different"
        view = h.driver._qualification_feedback(h.campaign,
            seed_identity="different" if changed == "source" else source)
        assert view["status"] == "unavailable" and REASON not in json.dumps(view)
    h.driver.state.data = original
    def missing_source(*args, **kwargs):
        raise FileNotFoundError("synthetic removed retained source")
    monkeypatch.setattr(h.driver.state, "read_source", missing_source)
    view = h.driver._qualification_feedback(h.campaign, seed_identity=source)
    assert view["status"] == "unavailable" and view["reason"] == "qualification-sealed-source-unavailable"
