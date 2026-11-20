# Fixed Schema backend

`aka.workflow.schema.backend:create_workflow` now implements A3's default
factory. It returns the retained Light Workflow and ManagedCampaign, runs A4's
exact-source transactions through the original LongHorizonCampaign, and never
calls a planner. `workflow_backend()` implements KS's synchronous `implement/aka`
entry using the same controller and original OwnerAccount.

## Configuration and entry points

Prepare an immutable KS implement request with the complete Task, fixed Schema,
scope, materials, and Rules containing `fidelity.freeze_method(publication)`.
The profile must freeze the explicit `parameters.claude` native CLI profile.
No token limit can be enforced by that native no-tools transport; finite
request/parent/native token limits are rejected. Call/job/active limits remain
with KS's original authority.

The optional selection retains the original A3 schema:

```json
{
  "format": "aka.schema.selection.v1",
  "request": "/absolute/bundle/request.json",
  "run_root": "/absolute/private/schema-run",
  "backend": {
    "factory": "aka.workflow.schema.backend:create_workflow",
    "implementation_sha256": "sha256:<installed backend.py digest>",
    "parameters": {}
  }
}
```

```sh
aka optimize --repo-root "$AKA" --problem /absolute/bundle \
  --platform H20 --framework triton --framework-baseline never \
  --convert-after 0 --token-budget 0 --schema \
  --schema-config /absolute/selection.json

kernelschema implement --request /absolute/bundle/request.json --out /absolute/private/run
kernelschema implement --continue /absolute/private/run
```

For the KS direction the request selects `backend.name="aka"`; AKA must already
be installed. KS only lazily discovers that named backend. It does not install
AKA, copy a parent account, or invoke a second top-level implement loop.
`profile.parameters.aka.native` optionally freezes `workspace`, `name`,
`settings` (original WorkflowSettings fields), and `runtime` (original
CampaignRuntimeSettings fields). Without it, the private native workspace is
`<run>/native`, using Triton/never/no conversion and 20 native versions. An
explicit request should supply its intended platform and native limits.

The AKA direction also retains the original effective native CLI settings in
its private state before fees. Resume requires those same settings, selection
bytes, KS release, service identity and native root. Missing run state cannot
create a replacement account or seed.

## Services and qualification

`profile.parameters.aka.services` selects an installed, source-hashed factory:

```python
{"factory": "aka.workflow.schema.agate:workflow_services",
 "implementation_sha256": "sha256:<installed agate.py digest>",
 "parameters": {"native_qualification": qualification_ref}}
```

This concrete factory consumes K6's existing
`profile.parameters.evaluation` selection and original Agate probe, baseline,
and four negative-control captures. It uses the existing official SDK config
through `KS_AGATE_CONFIG`; it does not find credentials, install a remote stack,
or submit qualification jobs. Missing auth/config is a missing prerequisite,
not proof that the service is unavailable.

AKA separately requires an original `aka.native.qualification.v1` artifact:
`profile=aka.schema.claude.no-tools.v1`, exact `claude_sha256`, exact source
hashes for `session`, `stdin_exec` and original `aka.agent.native`,
`synthetic=false`, and coder/reviewer entries containing an original
`NativeInvocation` Ref and A3 native locator. Admission rechecks the executable,
actual permission initialization, model/session, output parsing and complete
usage. K3 runtime qualification does not sign this AKA transport. This batch
has no actual native/model/GPU qualification capture.

Retain the qualification artifact and both original invocation Refs as explicit
Materials entries with `roles=[]`. KS imports that selected closure while no
model role receives it; profile parameters alone do not import nested Ref
files. No credential or whole prior run directory belongs in these materials.

An unconfigured installation returns a zero-fee `blocked/new_request` result.
It never maps LocalExecutor's CPU logic into CUDA PASS. Test-only injected
services exercise successful branches and are labeled synthetic in their raw
artifacts and test source.

## Operations, recovery and result

Every actual operation has a persisted run, role, exact C, e, reservation,
submission key, and original native locator. NativeSession retains the complete
NativeInvocation before returning to policy. A lost outer return reuses that
receipt and exact installed C. An invocation without a recoverable original
receipt remains `blocked/settle_only`; no new model, key or channel replaces it.
The original process/session/exec-intent and output directory stay available.

The only balance/clock implementation is OwnerAccount. V0, concurrent prefetch
reviewers, each coder/reviewer/repair, full Task evaluation, production review,
and comparison have separate receipts. V0 prefetch uses an atomic finite group;
group transfer uses the original `claim_group`. Native total token usage is
absorbed once, without adding its child events again. Unknown statistics stay
null. A native terminal says nothing about a sibling remote job's cleanup.

For this Schema profile, configured native prefetch labels identify advisory
review slots. Each slot uses the request's explicitly selected restricted
Claude reviewer profile; its label remains in the original operation purpose.
It is not a claim that a Codex/Qoder model was invoked. A different reviewer
transport requires its own explicit profile and implementation.

Authorization runs at actual Agent launch. `check_active` also runs while the
native process works, in the stdin shim immediately before exec, and before
the real evaluator/Agate submission. Absolute original deadlines narrow the
call after preparation; pause/fence does not grant a new interval. The parent
path is carried as `KS_BUDGET_AUTHORITY`, never as a mutable budget descriptor.

The public result follows KS's unchanged protocol. `code.latest`,
`code.qualified`, per-C assessments, `performance_goal`, native incumbent G,
and native promotion count are independent. A full Task/current-fidelity pass
can be completed while slower, unpromoted, or missing a native comparison
service. `verify_only` requires an explicit nonempty seed and has zero coder,
editing episodes or new promotions; evaluation/review remain paid operations.

## Native audits and comparison

The host uses the original CandidateWorkspace seal. Qualification is durable
before ABBA, then accompanies the native attempt and commit-bound promotion
audit. Attempt usage includes later comparison receipts separately from the
unchanged pre-ABBA qualification. Uncommitted recovery reuses that binding;
committed recovery absorbs the original audit/usage without another review or
promotion. A committed audit missing qualification preserves native history
and returns Schema unknown/blocked.

RecordedABBAValidator has a narrow explicit callback that avoids the legacy
Atrex-Bench argv construction for this Task. The fixed tree's base
GatewayABBAValidator only holds configuration; it has no independent `verify`
implementation. The retained ABBA schedule, score and promotion decision remain
the native implementations.

`ComparisonCallback` requires a service with `admission`, `locator`, `prepare`,
`validate`, `submit(..., authorize=...)`, `settle`, and `measurement` methods.
The original spec binds both sealed sources, Task/scope, schedule, timeout,
reservation and qualification digest. Its terminal must retain one allocation
ID, exact schedule, unchanged full Task report for every row, full coverage
measurements and the allocation's single cumulative usage receipt. It is not
permitted to relabel independent public timing or jobs from different
allocations as ABBA. CPU tests exercise this protocol against an explicit fake
allocation, including original native promotion and commit recovery. No actual
Agate same-allocation comparison has been qualified; the concrete Agate service
therefore leaves comparison unselected and can complete qualified unpromoted C.

## A6 seam

A6 can consume `FixedController.execute()` / `Backend.execute(context)` and the
unchanged public result. It must settle/close the old transaction, build a new
immutable request for changed R or mode, and connect its original parent via
`use_authority` or `KS_BUDGET_AUTHORITY`. Ordinary build, numerical or fidelity
defects continue the same R; the fixed controller never interprets them as
permission to plan. Planner scheduling and revised-request admission belong to
A6, not this backend.

The controller assumes the original native campaign writer lease, as the A1/A4
seams do. WorkflowStartup holds it on the AKA CLI path; `Backend.execute` holds
it for KS. An A6 caller already holding that lease should use its retained
controller instead of trying to acquire a second lease.
