# Schema source lifecycle (A4 CPU)

`aka.workflow.schema.preparation.SeedLifecycle` connects the existing A1
single-episode call to exact, persistent C. It consumes the installed fixed KS
release through `kernelschema.api.RunContext`, A2's source validation/binding,
and A3's `NativeRoleResult`/`NativeCallAuthority`. It creates no account, model
runtime, evaluator, planner loop, qualification or promotion policy.

```python
from aka.workflow.schema.preparation import SeedLifecycle

# run is the original KS RunContext; engine is the original native engine;
# owner is A5's adapter to the one existing KS OwnerAccount.
sources = SeedLifecycle.create(run, engine, owner)  # before any native work
outcome = sources.run_episode(1)
c1 = sources.current_source()  # SourceBinding, independently of native G

# Same native lock, original request/profile/budgets and native workspace.
sources = SeedLifecycle.reopen(run, engine, owner)
outcome = sources.run_episode(2)
```

Creation is explicit. Missing/corrupt state cannot fall through to creation,
and a native repository/active episode cannot acquire a newly guessed seed.
The private `aka-source-lifecycle` state uses KS's existing durable state API,
which binds run ID, frozen request digest and owner. It additionally freezes
Task/R/rules/scope/profile, source and native identities, native location,
framework/conversion policy, configured V0 reviewers and original limits.
Use it serially under the outer native workflow's original lock. There is no
cross-machine restore or concurrent-writer claim.

## V0, C0, G and HEAD

Native `InitialSource` must equal the verified `Task.baseline_source`: the
executable FP32 softmax plus copy implementation. `never` still runs native
V0, including configured reviewer prefetch. C0 is exclusively explicit
`request.seed`, or `Task.starter_source` when seed is null. Invalid explicit
seed never falls back. A legal null/null empty start is recognized by KS but
unsupported by this native profile and is rejected before any fee admission.
An unexecutable starter is a valid C0; it is never measured as V0.

Only `single_file`/UTF-8/`kernel.py`/0644 is supported. KS owns the manifest,
size/hash, byte encoding and full payload closure checks. CRLF, Unicode and
missing final LF survive the original CandidateWorkspace installation. A2's
new `read_native_source` exposes the same existing validation without publishing
another Source; `freeze_native_source` delegates to it. Auxiliary source files
and changed declarations are rejected. Public native harness/context and
`solution.json` are bound separately; scratch remains diagnostic, outside C.

The native worktree forks the current canonical HEAD. Its ancestry is retained
in the seed intent, while its incumbent source G has a separate SourceBinding.
Measurement/format/outcome metadata can advance HEAD without changing G.
The new worktree receives the entire selected C; no source overlay or fallback
is used. Each terminal episode captures exact C from its retained native draft
before notifying the external owner. The next episode receives that C even
when native status is rejected/invalid_handoff and G was not promoted.
Older terminal replays cannot rewind latest C.

## Durable boundaries

| Boundary | Recovery behavior |
| --- | --- |
| Native V0 recorded | Bind original version-record bytes, V0 source revision and baseline C. Missing/changed evidence fails before setup can repeat V0. An unknown V0 operation still requires A5's original operation fence/reconciliation. |
| Seed intent | Persist e, current HEAD/base, branch/worktree, exact seed, G and native metadata identity before installation. |
| Installing / ready | Reenter only while preparing; reinstall complete original seed, fsync actual kernel and directory, then verify bytes, mode and native identity. |
| Draft registered | Verify complete C0 before durable admitted. A1 retains preparing until this confirmation returns; no first invocation can precede it. |
| Admitted / exploring | Use retained draft/state. Never reinstall seed. Validate source/context and prevent native controller drift from triggering EpisodeWorkspace's automatic source replacement. A pause preserves this boundary. |
| Complete result installation interrupted | Reuse the original operation and publication ID. Before/after complete bytes can finish the same installation. Unknown partial or user-modified bytes fence and remain diagnostic; a new native invocation is blocked until original result installation is reconciled. |
| Native terminal / notification lost | Capture C under a durable publication ID, then record terminal and latest C before replayable owner notification. Missing/corrupt source fences and preserves the last complete C. |

`current_source(e)` returns the most recent complete stable C for e, including
when a later damaged draft is fenced. It does not claim that the damaged draft
equals that C or that C has passed evaluation/fidelity. `current_source()` is
the latest terminal C (C0 before the first terminal); an active accepted result
is available with the explicit episode argument.

## A5 result and fee interface

```python
# authority is the original NativeCallAuthority, with its existing reservation;
# result is returned by the actual A3 RoleSession under that authority.
binding = sources.install_result(e, result, authority=authority)
```

This verifies original run, operation/reservation, episode, coder role,
session, assignment, profile and input C against the durable invocation intent.
It uses `NativeRoleResult.native_source()` and `freeze_source(publication,
candidate_workspace)`, then the original CandidateWorkspace installer. Model
content is bytes for the constant logical `kernel.py`; a model-provided path
never becomes a host destination. The installation persists operation,
publication ID, intent digest, exact target SourceBinding and final installed
state. Replaying a completed installation returns that binding without writing
over later edits. A5 retains the full NativeInvocation and settles its original
usage even when permission/output validation or installation fails.

No call to `seal_candidate`, evaluation, ABBA, audit, promotion or receipt
absorption occurs here. A5 must gate every actual V0/model/reviewer/retry and
each of the three native audits with the same OwnerAccount; lifecycle
`admit(prepare/recover/episode)` remains only a coarse prelude boundary.
Original reservation, operation ID, input C, e and native locator precede the
call. Returned binding supplies public C digest, native identity, kernel digest,
size and mode for later seal/evaluation/audit receipts. Unknown execution,
pause, clock, finish-floor and receipt deduplication stay with that owner.

## Native preparation decision and validation scope

The supported fixed R0 path uses `framework_baseline="never"`, Triton and no
native stack conversion. `always`/`auto` explicitly reject: a restricted V1
with fixed-role context, fidelity and fee gates has not been qualified.
The native single-file format/measurement helpers remain in place. This Task's
direct evaluator does not require solution metadata; its presence/identity or
legal absence is explicit in the episode state. `never` creates no V1 baseline
pin. A profile requiring that pin or a different source format needs a separately
implemented and validated restricted V1 path.

Native `_finish` still checks/export G. A5 must separately check/stage exact C
for production/source/dependency qualification and publish its own completion,
soft-goal and qualified/latest facts. C preservation alone proves neither
native production delivery nor slow-but-qualified completion. V0 prefetch and
production reviewers retain real fee responsibilities even with `never`.

Tests use real A2 ProblemProvider/LegacySourceProvider, the original native
Store/GitCandidateWorkspace/EpisodeWorkspace/A1 loop, and an actual A3
Agent/NativeSession with a synthetic process. A synthetic CPU Evaluation
executes the real Task baseline and supplies labelled synthetic timing solely
to cross the native V0 boundary. Actual A2 CPU evaluation remains
`UNAVAILABLE`/`all_pass=false`; no production projection was changed. No model,
GPU, Agate, full A5 loop, actual ABBA/three-audit or production qualification is
claimed.
