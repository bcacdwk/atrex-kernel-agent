import base64
import json

import pytest

from aka.contracts.workspace import RevisionContent
from aka.task.candidate_workspace.provider import GitCandidateWorkspace
from aka.workflow.light.long_horizon.store import CampaignStore
from aka.workflow.schema.bridge import run_episode
from schema_cpu_fixture import BASELINE, Owner, make_engine


def test_exact_episode_replay_and_native_lifecycle(tmp_path):
    campaign, engine = make_engine(tmp_path)
    owner = Owner(tmp_path / "owner")
    result = run_episode(engine, 1, lifecycle=owner)
    assert result.reason == "episode-settled"
    assert result.attempt["status"] == "invalid_handoff"
    assert campaign.setup_count == 1 and campaign.finish_count == 0
    assert len(campaign.evaluations) == len(campaign.agent.calls) == 1
    assert campaign.evaluations[0][1] == campaign.agent.calls[0]["received"] == BASELINE
    assert (tmp_path / "owner/native-locator.json").is_file()
    assert not (tmp_path / "owner/result.json").exists()
    before = list(owner.events)
    replay = run_episode(engine, 1, lifecycle=owner)
    assert replay == result
    assert owner.events == before + [("settled", 1)]
    assert campaign.setup_count == 1 and len(campaign.evaluations) == len(campaign.agent.calls) == 1
    assert not (result.native_root / "episodes/e0002").exists()
    # The original outer Campaign remains the sole finalization owner.
    campaign._finish(result.reason)
    assert campaign.finish_count == 1


@pytest.mark.parametrize("seed", [b"raise NotImplementedError('starter')\n", b"# nonempty partial\ndef forward():\n    return None\n"])
def test_v0_seed_head_and_incumbent_are_distinct(tmp_path, seed):
    campaign, engine = make_engine(tmp_path)
    owner = Owner(tmp_path / "owner")
    preparation = []
    def prepare(*, worktree, active):
        assert active["phase"] == "preparing"
        assert len(campaign.evaluations) == 1
        assert not campaign.agent.calls
        assert campaign.candidate_workspace.current_revision(campaign.workspace) == worktree.base_commit
        preparation.append(worktree.base_commit)
        campaign.candidate_workspace.install_source(worktree.path, RevisionContent({"kernel.py": seed}))
    first = run_episode(engine, 1, lifecycle=owner, prepare_episode=prepare)
    v0 = json.loads((campaign.workspace / ".atrex_versions/v0.json").read_text())
    assert v0["git_commit_hash"] != preparation[0]  # Measurement/format metadata advances HEAD.
    assert campaign.candidate_workspace.read_source(campaign.workspace,
        revision=v0["git_commit_hash"]).files["kernel.py"] == BASELINE
    source = campaign.candidate_workspace.read_source(campaign.workspace)
    assert source.files["kernel.py"] == BASELINE
    assert campaign.evaluations[0][1] == BASELINE
    assert campaign.agent.calls[0]["received"] == seed
    new_head = campaign.candidate_workspace.current_revision(campaign.workspace)
    assert first.attempt["base_commit"] == preparation[0]
    assert first.attempt["outcome_commit"] == new_head != preparation[0]
    public = json.loads((first.native_root / "episodes/e0001/public-candidate.json").read_text())
    partial = base64.b64decode(public["files"]["kernel.py"]["content_base64"])
    assert partial == seed + b"\n# synthetic coder progress\n"
    def next_prepare(*, worktree, active):
        assert worktree.base_commit == new_head
        campaign.candidate_workspace.install_source(worktree.path, RevisionContent({"kernel.py": partial}))
    second = run_episode(engine, 2, lifecycle=owner, prepare_episode=next_prepare)
    assert second.attempt["base_commit"] == new_head
    assert campaign.agent.calls[1]["received"] == partial
    assert campaign.candidate_workspace.read_source(campaign.workspace).files["kernel.py"] == BASELINE
    assert len(campaign.evaluations) == 1 and campaign.setup_count == 1
    assert CampaignStore.checkpoint(campaign.workspace)[0].tokens == 14


@pytest.mark.parametrize("limit,reason", [(dict(max_version=0), "budget: max-iters"),
    (dict(token_budget=7), "budget: token-budget"),
    (dict(max_version=None, max_episodes=1), "max-episodes")])
def test_native_limits_prevent_more_calls(tmp_path, limit, reason):
    campaign, engine = make_engine(tmp_path)
    owner = Owner(tmp_path / "owner")
    run_episode(engine, 1, lifecycle=owner)
    for name, value in limit.items():
        setattr(engine, name, value)
    events = list(owner.events)
    result = run_episode(engine, 2, lifecycle=owner)
    assert result.reason == reason and result.attempt is None
    assert len(campaign.agent.calls) == len(campaign.evaluations) == 1
    assert owner.events == events


@pytest.mark.parametrize("phase", ["prepare", "episode"])
def test_external_denial_starts_no_coder_or_active_episode(tmp_path, phase):
    campaign, engine = make_engine(tmp_path)
    owner = Owner(tmp_path / "owner")
    owner.denied.add(phase)
    result = run_episode(engine, 1, lifecycle=owner)
    assert result.reason == f"external-admission: {phase}"
    assert not campaign.agent.calls and not owner.terminals
    assert len(campaign.evaluations) == (phase == "episode")
    assert not (result.native_root / "active_episode.json").exists()


def test_lost_terminal_return_recovers_exact_e_without_coder(tmp_path):
    campaign, engine = make_engine(tmp_path)
    engine.episode_limit = 1
    class LostReturn(GitCandidateWorkspace):
        armed = True
        def record(self, workspace, **kwargs):
            revision = super().record(workspace, **kwargs)
            if self.armed and kwargs["message"].startswith("v1: long-horizon episode"):
                self.armed = False
                raise KeyboardInterrupt("synthetic loss after native Git commit")
            return revision
    campaign.candidate_workspace = LostReturn()
    owner = Owner(tmp_path / "owner")
    with pytest.raises(KeyboardInterrupt):
        run_episode(engine, 1, lifecycle=owner)
    state, active = CampaignStore.checkpoint(campaign.workspace)
    assert state.tokens == 7 and active["phase"] == "recording"
    assert not owner.terminals
    resumed = run_episode(engine, 1, lifecycle=owner)
    assert resumed.attempt["recovered_after_supervisor_interruption"]
    assert resumed.attempt["status"] == "invalid_handoff"
    assert len(campaign.agent.calls) == len(campaign.evaluations) == 1
    assert list(owner.terminals) == [1]
    assert CampaignStore.checkpoint(campaign.workspace)[0].tokens == 7
    assert not (resumed.native_root / "episodes/e0002").exists()


def test_settled_notification_is_replayable_after_failure(tmp_path):
    campaign, engine = make_engine(tmp_path)
    owner = Owner(tmp_path / "owner")
    owner.settlement_failure = RuntimeError("synthetic owner unavailable")
    with pytest.raises(RuntimeError, match="owner unavailable"):
        run_episode(engine, 1, lifecycle=owner)
    assert CampaignStore.checkpoint(campaign.workspace)[1] is None
    result = run_episode(engine, 1, lifecycle=owner)
    assert result.attempt["tokens"] == 7
    assert len(campaign.agent.calls) == len(campaign.evaluations) == 1
    assert len(owner.terminals) == 1


def test_agent_interrupt_pause_and_resume_preserve_draft(tmp_path):
    campaign, engine = make_engine(tmp_path)
    owner = Owner(tmp_path / "owner")
    preparation = []
    def prepare(**kwargs):
        preparation.append(kwargs["worktree"].episode)
    campaign.agent.fail = KeyboardInterrupt("synthetic interrupted invocation")
    with pytest.raises(KeyboardInterrupt):
        run_episode(engine, 1, lifecycle=owner, prepare_episode=prepare)
    assert not owner.terminals
    assert CampaignStore.checkpoint(campaign.workspace)[1]["episode"] == 1
    owner.denied.add("recover")
    assert run_episode(engine, 1, lifecycle=owner, prepare_episode=prepare).reason == "external-admission: recover"
    assert len(campaign.agent.calls) == 1
    owner.denied.clear()
    result = run_episode(engine, 1, lifecycle=owner, prepare_episode=prepare)
    assert preparation == [1]
    assert campaign.agent.calls[1]["received"] == BASELINE + b"\n# synthetic coder progress\n"
    assert result.attempt and len(campaign.evaluations) == 1


def test_preparation_interrupt_reenters_before_draft_registration(tmp_path):
    campaign, engine = make_engine(tmp_path)
    owner = Owner(tmp_path / "owner")
    preparation = []
    def prepare(*, worktree, active):
        preparation.append(active["phase"])
        if len(preparation) == 1:
            raise KeyboardInterrupt("synthetic seed preparation crash")
    with pytest.raises(KeyboardInterrupt):
        run_episode(engine, 1, lifecycle=owner, prepare_episode=prepare)
    assert not campaign.agent.calls
    assert CampaignStore.checkpoint(campaign.workspace)[1]["phase"] == "preparing"
    owner.denied.add("episode")
    assert run_episode(engine, 1, lifecycle=owner, prepare_episode=prepare).reason == "external-admission: episode"
    assert CampaignStore.checkpoint(campaign.workspace)[1]["phase"] == "preparing"
    owner.denied.clear()
    run_episode(engine, 1, lifecycle=owner, prepare_episode=prepare)
    assert preparation == ["preparing", "preparing"]
    assert len(campaign.agent.calls) == len(campaign.evaluations) == 1


def test_v0_interruption_preserves_locator_and_owner_can_fence(tmp_path):
    campaign, engine = make_engine(tmp_path)
    owner = Owner(tmp_path / "owner")
    campaign.evaluation_failure = KeyboardInterrupt("synthetic V0 job unknown")
    with pytest.raises(KeyboardInterrupt):
        run_episode(engine, 1, lifecycle=owner)
    assert json.loads((owner.root / "native-locator.json").read_text())["workspace"] == str(campaign.workspace)
    assert not owner.terminals and not campaign.agent.calls
    owner.denied.add("prepare")
    result = run_episode(engine, 1, lifecycle=owner)
    assert result.reason == "external-admission: prepare" and len(campaign.evaluations) == 1


def test_never_still_runs_v0_and_configured_prefetch(tmp_path, monkeypatch):
    campaign, engine = make_engine(tmp_path)
    owner = Owner(tmp_path / "owner")
    reviews = []
    campaign.v1_ask_codex = True
    def fake_reviewer():
        assert owner.events[0] == ("admit", "prepare", 1)
        assert (owner.root / "native-locator.json").exists()
        reviews.append("synthetic source-only reviewer")
    # Preserve the real native prefetch scheduler; substitute only its role's
    # implementation, just as the fixture substitutes model/evaluator services.
    monkeypatch.setattr(campaign, "_ensure_framework_baseline_correctness_guidance", fake_reviewer)
    run_episode(engine, 1, lifecycle=owner)
    assert campaign.framework_baseline == "never"
    assert len(reviews) == len(campaign.evaluations) == 1
    assert campaign.agent.calls[0]["received"] == BASELINE


def test_invalid_or_lost_state_cannot_start_new_work(tmp_path):
    campaign, engine = make_engine(tmp_path)
    owner = Owner(tmp_path / "owner")
    with pytest.raises(ValueError, match="next episode"):
        run_episode(engine, 2, lifecycle=owner)
    assert not campaign.evaluations
    run_episode(engine, 1, lifecycle=owner)
    state_path = campaign.workspace / ".atrex_long_horizon/state.json"
    original = json.loads(state_path.read_text())
    state_path.write_text(json.dumps(dict(original, tokens=0)))
    with pytest.raises(ValueError, match="retained receipts"):
        run_episode(engine, 2, lifecycle=owner)
    state_path.write_text("{")
    with pytest.raises(ValueError):
        run_episode(engine, 1, lifecycle=owner)
    state_path.unlink()
    with pytest.raises(ValueError, match="accounting is missing"):
        run_episode(engine, 1, lifecycle=owner)
    assert len(campaign.agent.calls) == len(campaign.evaluations) == 1


def test_fresh_zero_version_allowance_does_not_run_v0(tmp_path):
    campaign, engine = make_engine(tmp_path, max_version=0)
    owner = Owner(tmp_path / "owner")
    result = run_episode(engine, 1, lifecycle=owner)
    assert result.reason == "budget: max-iters"
    assert not campaign.agent.calls and not campaign.evaluations and not owner.events


def test_different_active_episode_is_rejected_before_recovery(tmp_path):
    campaign, engine = make_engine(tmp_path)
    owner = Owner(tmp_path / "owner")
    campaign.agent.fail = RuntimeError("synthetic active e1")
    with pytest.raises(RuntimeError, match="active e1"):
        run_episode(engine, 1, lifecycle=owner)
    before = list(owner.events)
    with pytest.raises(ValueError, match="different native episode"):
        run_episode(engine, 2, lifecycle=owner)
    assert owner.events == before and len(campaign.agent.calls) == len(campaign.evaluations) == 1
