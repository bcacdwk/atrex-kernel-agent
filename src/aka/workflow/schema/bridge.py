"""One specified episode through the existing native supervisor.

This module neither builds a Campaign nor owns its setup/finalization. The
outer workflow builds one engine with the frozen native version/token limits,
calls it for explicit episode numbers, and owns Campaign._finish once at the
actual end of that workflow. KS contracts and the parent's ledger live outside
this bridge.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, TYPE_CHECKING

from aka.workflow.light.long_horizon.store import CampaignStore

if TYPE_CHECKING:
    from aka.workflow.light.long_horizon.campaign import LongHorizonCampaign


class EpisodeLifecycle(Protocol):
    """Synchronous callbacks into the sole external budget owner.

    ``admit`` must durably bind the owner's run/operation key to ``workspace``
    (including its private .atrex_long_horizon root) before returning True.
    It runs before prepare (V0/V1/prefetch/production), before native recovery
    when active state exists, and before starting/resuming the requested e.
    False forbids that step. Each stage can contain several paid operations;
    this is a lifecycle reservation seam, not per-model/job enforcement.

    The owner must also gate actual Agent/Evaluation calls and retries, keep
    pause/fence and clock state, and reconcile interrupted work. Recovery may
    evaluate/review, so its admission is not permission for free replay. No
    failed/raised admission is treated as a released reservation here.

    ``settled`` means only that the original native episode reached its durable
    terminal checkpoint and cleared active state. Native status/tokens do not
    assert that every remote job or unknown statistic has settled. This callback
    is replayable (including after its own failure); deduplicate actual receipts
    under the original owner operation key. Exceptions propagate, without a
    fabricated settlement, cancellation, refund, or automatic next episode.
    """

    def admit(self, *, phase: str, episode: int, workspace: Path,
              active: dict[str, Any] | None) -> bool: ...

    def settled(self, *, episode: int, workspace: Path,
                attempt: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    reason: str
    native_root: Path
    attempt: dict[str, Any] | None


def run_episode(engine: LongHorizonCampaign, episode: int, *,
                lifecycle: EpisodeLifecycle, prepare_episode: Callable | None = None) -> EpisodeResult:
    """Run/recover exactly e, or return its previously archived native outcome.

    The prepare callback is an A4 seam, called after native worktree creation
    and runtime linking, before the first Git-free Agent draft is registered.
    It receives ``worktree`` and a copy of the durable ``active`` mapping. It
    must own its intent/ready records and full-source validation, and be safe
    to reenter only while native phase is preparing. Exploring drafts are never
    reseeded. This bridge provides neither a seed installer nor recovery of an
    external parent's lost authorization.
    """
    reason = engine.run(episode=episode, lifecycle=lifecycle, prepare_episode=prepare_episode)
    state, active = CampaignStore.checkpoint(engine.workspace)
    attempt = next((dict(item) for item in state.attempts if item["episode"] == episode), None)
    if active is not None:
        attempt = None
    return EpisodeResult(episode, reason, engine.workspace.resolve() / ".atrex_long_horizon", attempt)
