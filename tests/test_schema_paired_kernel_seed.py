"""Paired E/D seeds through real CPU native owners; no model/GPU calls."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from aka.contracts.evaluation import VerificationResult
from aka.workflow.light.long_horizon.session import LongSessionRunner
from aka.workflow.light.long_horizon.store import CampaignStore
from aka.workflow.schema.materials import read_design
from aka.workflow.schema.run_state import RecoveryRequired
from test_schema_execution_context import files, output
from test_schema_native_execution_context import integrated, _coder, _seal, state_design
from test_schema_qualification_protocol import review


def rewrite(workspace):
    result = output(workspace, files(corrected=True, rewrite=True))
    path = workspace / "output/decision.json"
    decision = json.loads(path.read_text())
    decision["hypothesis_key"] = "paired-execution-rewrite"
    path.write_text(json.dumps(decision))
    return result


def originals(h):
    paths = [*h.driver.directory.joinpath("operations").rglob("*"),
             *h.driver.directory.joinpath("sources").rglob("*"),
             h.campaign.workspace / ".atrex_long_horizon/episodes/e0001/attempt.json"]
    return {path: path.read_bytes() for path in paths if path.is_file()}


@pytest.mark.parametrize("verdict", ["pass", "fail", "unknown"])
def test_rewrite_continues_sealed_unpromoted_kernel_with_fresh_review(integrated, monkeypatch, verdict):
    h = integrated
    h.enable()
    h.actions.append(lambda w: output(w, files(corrected=True)))
    h.driver._plan(h.campaign)
    _seal(monkeypatch, h.campaign)
    verified = []
    def verify(workspace, **kwargs):
        # Execute the CPU source; the synthetic slower timing keeps the native
        # champion unchanged after a real independent Schema pass.
        h.campaign.measure_for_acceptance(workspace, ())
        verified.append((workspace / "kernel.py").read_bytes())
        return VerificationResult("FAIL", 12.0, 10.0, -20.0)
    monkeypatch.setattr(h.campaign, "acceptance_verifier", lambda **kwargs:
        SimpleNamespace(repeats=1, per_run_timeout=30, shape_batch_size=1, verify=verify))
    h.actions.extend([_coder, lambda w: review(w, verdict=verdict)])
    h.driver._implement(h.campaign)
    old = deepcopy(h.driver.state.data)
    first = old["episodes"]["1"]
    attempt = CampaignStore.checkpoint(h.campaign.workspace)[0].attempts[0]
    assert attempt["candidate_qualification"]["qualified"] == (verdict == "pass")
    assert first["qualification_feedback"]["verdict"] == verdict and not first["accepted"]
    assert bool(verified) == (verdict == "pass")
    draft = old["working_draft"]
    assert draft["identity"] == first["sealed_candidate"]["identity"] != old["champion"]["identity"]
    source = h.driver.state.read_source(draft, owner=h.campaign.candidate_workspace)
    retained = originals(h)
    h.driver._close_window()
    history = deepcopy(h.driver.state.data["hypotheses"])
    h.actions.append(rewrite)
    h.driver._plan(h.campaign)
    state = h.driver.state.data
    assert state["planner_request"]["replacement_seed_role"] == "working_draft"
    assert state["hypotheses"][state["hypothesis"]]["seed"] == state["working_draft"] == draft
    assert state["champion"] == old["champion"]
    assert read_design(state_design(state)).manifest["parent"] == old["design"]
    assert state["hypotheses"][old["hypothesis"]] == history[old["hypothesis"]]
    assert h.driver._qualification_feedback(h.campaign, seed_identity=draft["identity"])["status"] == "unavailable"
    observed = []
    run = LongSessionRunner._run
    def actual_entry(self, workspace, prompt, **kwargs):
        binding = deepcopy(h.driver.state.data["binding"])
        assert (workspace / "kernel.py").read_bytes() == source.files["kernel.py"]
        assert binding["seed"] == draft and binding["design"] == state["design"]
        assert binding["context"] in prompt
        assert read_design(state_design(state)).checked.files["root.yaml"].decode() in prompt
        feedback = json.loads(binding["context"].split("Unavailable feedback does not establish fidelity.\n", 1)[1])
        assert feedback["status"] == "unavailable"
        observed.append(binding)
        return run(self, workspace, prompt, **kwargs)
    monkeypatch.setattr(LongSessionRunner, "_run", actual_entry)
    h.actions.extend([_coder, lambda w: review(w, verdict="fail")])
    h.driver._implement(h.campaign)
    state = h.driver.state.data
    second = state["episodes"]["2"]
    native = CampaignStore.checkpoint(h.campaign.workspace)[0].attempts[-1]
    assert len(observed) == 1 and second["binding"] == observed[0]
    assert native["episode_inputs"]["seed_identity"] == draft["identity"]
    assert native["episode_inputs"]["qualification_required"] is True
    assert second["qualification_feedback"]["verdict"] == "fail" and not second["accepted"]
    q = h.calls[-1] / "inputs"
    assert (q / "initial-seed/kernel.py").read_bytes() == source.files["kernel.py"]
    assert json.loads((q / "review/binding.json").read_text())["design_identity"] == state["design"]["identity"]
    assert state["episodes"]["1"] == first
    assert all(path.read_bytes() == raw for path, raw in retained.items())
    assert all(state["calls"][key] == value for key, value in old["calls"].items())
    assert all(state["usage_receipts"][key] == value for key, value in old["usage_receipts"].items())
    assert state["counts"]["agent_calls"] == 6 and state["counts"]["tokens"] == 108
    assert state["counts"]["dispatches"] == old["counts"]["dispatches"]
    assert state["prepared"] == old["prepared"] and state["root_ref"] == old["root_ref"]


@pytest.mark.parametrize("seed_case", ["draft", "no-draft", "legacy"])
def test_lost_e_return_and_pending_d_replay_keep_frozen_seed(integrated, monkeypatch, seed_case):
    h = integrated
    h.enable()
    h.actions.append(lambda w: output(w, files(corrected=True)))
    h.driver._plan(h.campaign)
    h.actions.append(_coder)  # Actual unqualified native invalid_handoff draft.
    h.driver._implement(h.campaign)
    old = deepcopy(h.driver.state.data)
    draft, champion = old["working_draft"], old["champion"]
    assert draft["identity"] != champion["identity"]
    h.driver._close_window()
    if seed_case == "no-draft":
        h.driver.state.data["working_draft"] = None
    optimize = h.driver.planner.optimize
    def lost(request):
        if seed_case == "legacy":
            # Model an operation frozen by the previous release before its
            # native intent/publication; it already has original source bytes.
            h.driver.state.data["planner_request"].pop("replacement_seed_role")
            h.driver.state.save()
        optimize(request)
        raise RuntimeError("synthetic E return lost after publication")
    monkeypatch.setattr(h.driver.planner, "optimize", lost)
    h.actions.append(rewrite)
    with pytest.raises(RuntimeError, match="E return lost"):
        h.driver._plan(h.campaign)
    h.driver.state.data = json.loads(h.driver.state.path.read_bytes())
    frozen = deepcopy(h.driver.state.data["planner_request"])
    paid = deepcopy(h.driver.state.data["counts"])
    retained = originals(h)
    # Recovery must not choose from mutable current pointers after the E input
    # was frozen, whether the original choice was draft or fallback champion.
    h.driver.state.data["working_draft"] = champion if seed_case == "draft" else draft
    h.driver.state.save()
    monkeypatch.setattr(h.driver.planner, "optimize", optimize)
    monkeypatch.setattr(h.driver, "_new_context", lambda *a: pytest.fail("replay discovered new context"))
    h.driver._plan(h.campaign)
    state = h.driver.state.data
    expected = draft if seed_case == "draft" else champion
    assert state["planner_request"] == frozen and state["counts"] == paid
    assert state["hypotheses"][state["hypothesis"]]["seed"] == state["working_draft"] == expected
    assert all(path.read_bytes() == raw for path, raw in retained.items())
    engine = h.campaign.create_engine
    def interrupted(**kwargs):
        assert h.campaign.candidate_workspace.source_identity(kwargs["episode_seed"]) == expected["identity"]
        raise RuntimeError("synthetic D stopped after durable binding")
    monkeypatch.setattr(h.campaign, "create_engine", interrupted)
    with pytest.raises(RuntimeError, match="D stopped"):
        h.driver._implement(h.campaign)
    h.driver.state.data = json.loads(h.driver.state.path.read_bytes())
    binding = deepcopy(h.driver.state.data["binding"])
    assert binding["seed"] == expected
    h.driver.state.data["working_draft"] = champion if seed_case == "draft" else draft
    h.driver.state.save()
    monkeypatch.setattr(h.campaign, "create_engine", engine)
    def coder(workspace):
        assert (workspace / "kernel.py").read_bytes() == h.driver.state.read_source(
            expected, owner=h.campaign.candidate_workspace).files["kernel.py"]
        return _coder(workspace)
    h.actions.append(coder)
    h.driver._implement(h.campaign)
    state = h.driver.state.data
    assert state["episodes"]["2"]["binding"] == binding
    assert state["episodes"]["1"] == old["episodes"]["1"]
    assert all(path.read_bytes() == raw for path, raw in retained.items())
    native = CampaignStore.checkpoint(h.campaign.workspace)[0].attempts[-1]
    assert native["episode_inputs"]["seed_identity"] == expected["identity"]
    assert state["counts"]["agent_calls"] == 4 and state["counts"]["tokens"] == 72


@pytest.mark.parametrize("damage", ["reference", "bytes", "missing"])
def test_invalid_working_draft_is_rejected_before_new_planner_call(integrated, damage):
    h = integrated
    h.enable()
    h.actions.append(lambda w: output(w, files(corrected=True)))
    h.driver._plan(h.campaign)
    h.actions.append(_coder)
    h.driver._implement(h.campaign)
    h.driver._close_window()
    state = h.driver.state.data
    draft = deepcopy(state["working_draft"])
    path = h.driver.directory / draft["object"]
    if damage == "reference":
        draft["object"] = "../untrusted.json"
        state["working_draft"] = draft
    elif damage == "bytes":
        value = json.loads(path.read_bytes())
        value["files"]["kernel.py"] = "cGFzcwo="
        path.write_text(json.dumps(value))
    else:
        path.unlink()
    paid, calls = deepcopy(state["counts"]), len(h.calls)
    with pytest.raises((RecoveryRequired, FileNotFoundError)):
        h.driver._plan(h.campaign)
    assert state["counts"] == paid and len(h.calls) == calls
    assert state["design_version"] == 1
