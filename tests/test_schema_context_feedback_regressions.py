"""Regression for independently reproduced native same-source feedback masking."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
import pytest
from aka.contracts.workspace import CandidateHandle
from aka.workflow.light.long_horizon.campaign import LongHorizonCampaign
from aka.workflow.light.long_horizon.models import EpisodeHandoff
from aka.workflow.light.long_horizon.session import LongSessionRunner
from aka.workflow.light.long_horizon.store import CampaignStore
from aka.workflow.schema.run_state import RunLimits
from test_schema_qualification_feedback import projection, REASON, sha
from test_schema_workflow import controller, native
from aka.workflow.schema.qualification_feedback import applies_to, project, MAX_CHARACTERS
from aka.workflow.schema.workflow import SchemaWorkflow, build_feedback


def test_selector_skips_newer_wrong_context_and_renderer_only_uses_explicit_summary(projection):
    receipt, kwargs = projection
    summaries = []
    for number, context in ((1, "a" * 64), (2, "b" * 64)):
        kwargs["binding"].update(episode=number, execution_context_sha256=context)
        receipt["binding"]["execution_context_sha256"] = context
        summaries.append(project(receipt, **kwargs))
    state = SimpleNamespace(data={"prepared": kwargs["prepared"], "design": kwargs["binding"]["design"],
        "execution_context": {"execution_context_sha256": "a" * 64},
        "hypothesis": None, "hypotheses": {}, "champion": None, "tested_candidate": None,
        "working_draft": None, "diagnostics": [],
        "episodes": {str(summary["original_episode"]): {"episode": summary["original_episode"],
            "binding": {"design": kwargs["binding"]["design"]},
            "sealed_candidate": {"identity": "sealed"}, "qualification_feedback": summary}
            for summary in summaries}})
    original = deepcopy(state.data)
    driver = object.__new__(SchemaWorkflow)
    driver.state = state
    for seed in (None, "sealed"):
        selected = driver._qualification_feedback(None, seed_identity=seed)
        assert selected["status"] == "reviewed" and selected["original_episode"] == 1
    rendered = json.loads(build_feedback(state, qualification_feedback=selected))["qualification_feedback"]
    assert rendered["original_episode"] == 1 and rendered["scope"] == "historical-reviewed-source"
    assert json.loads(build_feedback(state))["qualification_feedback"]["status"] == "unavailable"
    wrong = json.loads(build_feedback(state, qualification_feedback=summaries[1]))["qualification_feedback"]
    assert wrong["reason"] == "qualification-binding-mismatch"
    # D must use its actual frozen binding, even if the state context differs.
    assert driver._qualification_feedback(None, seed_identity="sealed",
        context_binding={"execution_context_sha256": "b" * 64})["original_episode"] == 2
    assert state.data == original


def test_later_unreviewed_same_source_keeps_previous_failure_at_next_native_prompt(tmp_path, monkeypatch, projection):
    campaign = native(tmp_path)
    sealed, receipts = {}, []
    receipt, _ = projection
    def qualify(*, binding, source, source_identity, **kwargs):
        value = deepcopy(receipt)
        value["source_identity"] = source_identity
        value["operation_id"] = binding["operation_id"] + ":qualification"
        value["qualified"] = False
        value["substantive_change"] = True
        value["binding"].update(prepared_id=binding["prepared_id"],
            design_identity=binding["design"]["identity"], source_identity=source_identity,
            public_task_sha256=sha(driver.state.data["prepared"]["public_task"]),
            target_hardware_sha256=sha(driver.state.data["prepared"]["hardware"]),
            sealed_files={name: sha(raw) for name, raw in source.files.items()})
        value["evidence"][0]["evidence"] = [{"source": "candidate/kernel.py", "start": 1,
            "end": 1, "sha256": sha(source.files["kernel.py"])}]
        receipts.append(value)
        return value
    driver, _ = controller(tmp_path, campaign, limits=RunLimits(max_episodes=3), candidate_qualification=qualify)
    original_run, original_facts = LongSessionRunner.run, campaign.episode_facts
    def ready(self, workspace, *args, **kwargs):
        result = original_run(self, workspace, *args, **kwargs)
        _, active = CampaignStore.checkpoint(campaign.workspace)
        number = active["episode"]
        if number not in (1, 2):
            return result
        handle = CandidateHandle(number, active["base_commit"], active["episode_branch"], type(campaign.workspace)(active["worktree"]))
        if number == 2:
            (workspace / "kernel.py").write_bytes(sealed[1].files["kernel.py"])
        source = campaign.candidate_workspace.read_source(workspace)
        commit = campaign.candidate_workspace.seal_candidate(handle, source)
        sealed[number] = source
        return replace(result, exit_status=0, timed_out=False,
            handoff=EpisodeHandoff("candidate_ready", candidate_commit=commit), completion_diagnosis="")
    def facts(episode):
        value = original_facts(episode)
        if episode in sealed:
            value["candidate"] = sealed[episode]
        return value
    monkeypatch.setattr(LongSessionRunner, "run", ready)
    monkeypatch.setattr(campaign, "episode_facts", facts)
    monkeypatch.setattr(LongHorizonCampaign, "_completion_check", lambda self, handle, *args:
        "synthetic native completion failure before qualification" if handle.episode == 2 else "")
    monkeypatch.setattr("aka.workflow.light.long_horizon.main_adapter.candidate_policy_violations", lambda *args, **kwargs: [])
    assert driver.run(campaign) == "budget: episodes"
    rows = driver.state.data["episodes"]
    assert rows["1"]["sealed_candidate"]["identity"] == rows["2"]["sealed_candidate"]["identity"]
    assert rows["2"]["sealed_candidate"]["identity"] == rows["3"]["binding"]["seed"]["identity"]
    assert rows["1"]["qualification_feedback"]["verdict"] == "fail"
    assert rows["2"]["qualification_feedback"]["reason"] == "qualification-receipt-missing"
    assert len(receipts) == 1
    assert REASON in campaign.agent.calls[1]["prompt"]
    assert REASON in campaign.agent.calls[2]["prompt"]
    assert driver._qualification_feedback(campaign, seed_identity=rows["3"]["binding"]["seed"]["identity"])["original_episode"] == 1


@pytest.mark.parametrize("bound,reviewed,expected", [
    (None, None, "legacy"), ("a" * 64, "a" * 64, "bound"),
    (None, "a" * 64, "unavailable"), ("a" * 64, None, "unavailable"),
    ("a" * 64, "b" * 64, "unavailable"), ("invalid", "invalid", "unavailable")])
def test_feedback_context_requires_exact_native_binding(projection, bound, reviewed, expected):
    receipt, kwargs = projection
    if bound is not None:
        kwargs["binding"]["execution_context_sha256"] = bound
    if reviewed is not None:
        receipt["binding"]["execution_context_sha256"] = reviewed
    receipt["evidence"][0]["reason"] *= 500
    summary = project(receipt, **kwargs)
    assert len(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"))) <= MAX_CHARACTERS
    if expected == "unavailable":
        assert summary["status"] == "unavailable"
    else:
        assert summary["execution_context_status"] == expected
        assert applies_to(summary, prepared=kwargs["prepared"], design_identity="design", execution_context_sha256=bound)
        if bound is not None:
            assert not applies_to(summary, prepared=kwargs["prepared"], design_identity="design", execution_context_sha256=None)
