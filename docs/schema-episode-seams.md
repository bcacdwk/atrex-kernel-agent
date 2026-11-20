# Schema episode integration seams (A1 CPU)

This is an engineering interface, exercised with synthetic CPU services. It
does not enable the Schema switch, install KS, or qualify a GPU/backend.

`aka.workflow.schema.bridge.run_episode(engine, e, lifecycle=owner,
prepare_episode=prepare)` calls the existing `LongHorizonCampaign.run` with an
explicit positive episode number. It returns `EpisodeResult(episode, reason,
native_root, attempt)`. `attempt=None` means no complete native terminal was
returned. A native terminal is not a Schema completion/qualification result.

The caller supplies one existing engine with the frozen native `max_version`
and `token_budget` (or its original `max_episodes` mode). The bridge never
changes those allowances. Native zero `token_budget` still means unlimited;
an external zero/pause/fence must be denied by the owner.

The exact episode is either the next episode, the active episode, or an already
terminal episode. Another active e, skipped e, corrupt checkpoint, missing
accounting after episode work, and accounting below retained receipts fail
closed. An already terminal e is returned before setup/recovery and never
opens e+1. Native recovery that finishes e also stops immediately, including
when `episode_limit=1` would otherwise admit a further episode.

Lifecycle remains native. `LongHorizonCampaign.run` owns `prepare_campaign`
and its V0/resume/V1 prelude. The outer workflow owns finalization via the
original Campaign, once at its actual end; the bridge never calls `_finish`.
Do not call the whole `Campaign.run` for each e: its final production check can
perform review. A3/A5 should reuse the existing engine assembly when wiring
their outer owner.

| Seam | Timing and contract |
| --- | --- |
| `owner.admit(phase="prepare", episode, workspace, active)` | Before V0/resume preparation. Persist owner run/operation/native-root/version bindings and reservation before returning true. Preparation can include V0, V1, parallel reviewer prefetch, and production review. |
| `owner.admit(phase="recover", …)` | Before native active-episode recovery. Recovery may re-evaluate/review; it is not assumed free. |
| `owner.admit(phase="episode", …)` | After native upper-bound checks, before creating a fresh active e or resuming its coder. `active` is present for resume. False starts no new coder. |
| `prepare_episode(worktree, active)` | After a fresh worktree is linked and checked clean, before its first Git-free draft registration. Also reenters a preparing worktree after interruption. Exploring drafts never invoke it. Its durable preparing phase survives an intervening denial. |
| `owner.settled(episode, workspace, attempt)` | Only after native terminal archive/state is durable and active is cleared, or on replay of that terminal. Callback exceptions propagate; replay retries notification without another episode. |

These are synchronous lifecycle reservations/notifications, not a per-call
budget implementation. A5 must still gate the actual Agent/Evaluation/retry
operations, reserve minimum finishing work, and reconcile every raw receipt,
unknown statistic, pause/fence, and owner clock. Native terminal notification
does not settle an unknown remote job. Interrupted callbacks/invocations do
not release reservations or fabricate settlement. Deduplication belongs to
the same external owner and its original operation key.

`workspace` is available before the first fee/result, and locates the private
`.atrex_long_horizon` root, active marker, state, e archive, and usage receipts.
The CPU fixture proves this pre-return locator can be persisted and that a
lost native terminal return is recoverable. It does not implement a durable
external-process launch/lookup protocol. Missing external owner/version/run
bindings and unknown V0 jobs must remain fenced by A5; without that owner,
native V0 retry is not guaranteed free or deduplicated.

V0 and candidate source are separate. The real native `never` decision skips
V1, not V0; configured correctness reviewers can still prefetch during V0.
The CPU fixture executes a correct small baseline using a fake evaluator,
then installs an unexecutable starter or a nonempty partial candidate only at
the preparation seam. Synthetic timings do not qualify the actual Task.

The next e forks the current canonical HEAD. Native outcome/measurement/format
metadata can advance that HEAD even without a promotion or source change.
The incumbent source G can remain the V0 source while the archived unpromoted
candidate differs. A4 must install exact terminal C on the new HEAD ancestry,
with its own intent/ready/admitted records and frozen callback/seed identity;
it must not infer C from G or from Git HEAD alone. The fixture uses the actual
`public-candidate.json` bytes to continue C, but its installer is only a test,
not the A4 source/format/recovery protocol. Existing post-commit native recovery
may produce a sparse attempt; missing fields are not zero usage or evidence
of missing source. A5 must read the retained native usage/output artifacts.

The supported seam is serial use under the outer workflow's original native
lock and frozen configuration. No concurrent engine, cross-machine restore,
hidden model-session migration, complete V1/production/audit integration, or
external process recovery capability is claimed here.

Validation: `PYTHONPATH=src python -m pytest -q tests` in an isolated environment.
`tests/schema_off_probe.py` also runs the real default `Campaign.run` with
optional module imports, KS checkout I/O, and networking forbidden. Its
normalized prompts, sources, limits/outcomes, exports, setup and finish counts
can be compared against the recorded upstream source using `PYTHONPATH`.

Source provenance: `bridge.py`, namespace and tests are new native code. The
only existing source modifications are the narrow optional seams in
`light/long_horizon/campaign.py` and the explicit-episode checkpoint reader in
`light/long_horizon/store.py`, based on upstream
`bf5557ac04f6ff9988901d511cf25e1cbe61a1bc`. No external patch was migrated.
