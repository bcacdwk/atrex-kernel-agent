"""Explicit closed-stage boundary over the original native repository."""
import json

from kernelschema.api import open_run
from kernelschema.budget import OwnerAccount
from kernelschema.contracts import ProtocolError, digest
from aka.workflow.light.long_horizon.store import CampaignStore


def close_stage(context, campaign):
    state = context.load_state("aka-fixed-backend")
    owner = OwnerAccount.for_context(context, create=False)
    if state["phase"] != "done" or not state["finished"]:
        raise ProtocolError("original fixed finishing must settle before stage handoff")
    if owner.fences() or any(not (op["terminated"] or op["released"])
                             for op in owner.operations(context.run_id).values()):
        raise ProtocolError("original fixed operations must settle before stage handoff")
    native, active = CampaignStore.checkpoint(campaign.workspace)
    if active is not None:
        raise ProtocolError("original native episode must settle before stage handoff")
    owner.end(context.run_id)
    head = campaign.candidate_workspace.current_revision(campaign.workspace)
    source = context.load_state("aka-source-lifecycle") if head and context.request["mode"] == "code" else None
    if source and source["fence"]:
        raise ProtocolError("original native source is fenced")
    return {"run_root": str(context.root), "run_id": context.run_id,
            "request_sha256": context.request_sha256, "schema": context.request["schema"],
            "workspace": str(campaign.workspace.absolute()), "head": head,
            "episodes": native.episodes, "native_checkpoint": digest(json.dumps(native.as_dict(), sort_keys=True))
            if hasattr(native, "as_dict") else digest(json.dumps(native.__dict__, sort_keys=True)),
            "source_state": digest(source) if source else None,
            "native": state["binding"]["native"], "owner": str(owner.path),
            "latest": state["latest"], "qualified": state["qualified"]}


def validate_handoff(value, context, campaign):
    previous = open_run("implement", value["run_root"])
    if previous.request["task"] != context.request["task"]:
        raise ProtocolError("native handoff cannot change the complete Task")
    if previous.owner != context.owner:
        raise ProtocolError("native handoff cannot change owner")
    current_owner = OwnerAccount.for_context(context, create=False)
    if str(current_owner.path) != value["owner"]:
        raise ProtocolError("native handoff cannot allocate another balance")
    actual = close_stage(previous, campaign)
    if actual != value:
        raise ProtocolError("closed native stage changed before new source admission")
    # The old request/profile stays fixed. All original native constraints carry
    # forward, including cumulative max version and stall state in CampaignStore.
    for key, expected in value["native"].items():
        actual = getattr(campaign, key)
        if key == "min_improvement_pct":
            actual = str(actual)
        if actual != expected:
            raise ProtocolError("native handoff changed original limit/setting: " + key)
