# Schema workflow

Install matching `atrex-aka-schema`, `atrex-aka-light`, `atrex-aka-legacy` and
`atrex-aka-framework` distributions. Schema owns design, extract, optimize and
the fixed-design implementation driver. The selected native Agent and Executor
still own model calls, correctness, measurement and promotion. Planner designs
and revises Schema (C/E); Coder faithfully implements each frozen design (D) in
the normal Light Episode engine. Kernel and Schema evolve together across these
windows; correctness, design fidelity and champion promotion remain separate gates.

KernelSchema is not a runtime, build or submodule dependency. The migrated
language, specification, starter and selected Wiki are installed package data.
Their fixed source hashes and attribution remain in `assets/SOURCES.json` and
`NOTICE.md`; the upstream snapshot had no root license declaration, and AKA does
not claim to grant a new license for those portions.

Build all four distributions from this same source revision:

```sh
python -m pip install build 'packaging>=23' 'PyYAML>=6'
python tools/build_distributions.py --output /absolute/fresh-distributions \
  atrex-aka-framework atrex-aka-legacy atrex-aka-light atrex-aka-schema
python -m pip install --no-deps --force-reinstall /absolute/fresh-distributions/*.whl
```

Use a dedicated environment with the normal authenticated Agent CLI and native
execution prerequisites. Package version `0.2.0` alone does not identify the
source revision; retain the commit and wheel hashes for each real run.

## Start

Use the normal installed CLI with an explicit Schema configuration:

```sh
aka workflow --repo-root /absolute/public-inputs --profile light-sandbox \
  --patch /absolute/runtime.patch.json \
  --schema --schema-config /absolute/schema-config.json \
  --problem /absolute/atrex-bench/data/my_problem \
  --workspace-base /absolute/private-runs --workspace-suffix schema_run \
  --platform cuda --framework triton --agent-cli claude --hardware L20D \
  --framework-baseline never --convert-after 0 --fast-episodes 0 --max-iters 0
```

An embedding launcher uses the same assembly:

```python
from aka.bootstrap.host import run_selection
from aka.contracts.startup import Invocation
from aka.contracts.workflow_launch import encode_launch

run_selection(selection, Invocation(encode_launch(workflow_launch)),
              schema="/absolute/schema-config.json")
```

`selection` is the normal resolved AKA profile with the chosen Agent, Executor
and private environment. Schema does not start a separate model CLI or read
another repository's configuration. See the complete, non-sensitive
[L20D launch examples](../examples/schema/l20d/README.md).
For an already frozen Python `WorkflowLaunch`, explicitly set
`framework_baseline="never"`, `convert_after=0`, no baseline reviewers and a
`setup_timeout` within the Schema per-call limit; continuous operation also
needs `max_iters=0`. CLI fields omitted from a new request receive those defaults.

The configuration has `format: "aka.schema.launch.v1"`. Supply `initial` only
for a verified same-target kernel and complete Schema pair; its exact source
file hashes and retained evidence are checked against native preparation before
the first optimize call. The native `source` plugin selects those kernel bytes.
Without `initial` or `sota`, preparation performs fresh V0 and starts design.
Only an explicit source-only `sota` descriptor requests six-stage extraction.
The two L20D examples do not use extraction. A reference implementation is not
implicitly a SOTA candidate.

Relative Schema and evidence paths are relative to the configuration file.
Use a full directory containing `root.yaml`, `dispatch.md` and every referenced
node. A single node from a larger design is not a complete design. The
vector-add example retains an explicitly adapted historical proposal and uses
`skip_hardware: true`; that diagnostic does not replace independent target
identity, native correctness or current GPU measurement.

Omitting `--schema` leaves normal Light behavior unchanged and does not load the
Schema parser or assets. Workspaces cannot switch between modes; resume their
original controller or choose a new workspace.

## Frozen execution facts and Contract corrections

New native Schema operations receive a bounded execution context bound to the
prepared Task, complete public Task, target hardware and native policy hashes.
The selected framework comes from that frozen native policy. Architecture and toolchain
facts come from an applicable native preparation snapshot, an owner-verified
retained Job, or the selected native gateway's GET-only environment discovery.
SSH reuses preparation observations; Schema does not launch another remote probe.
Each fact retains its `observed`, `configured`, `declared`, `cache` or `unknown`
source and evidence hash. A GPU name, `campaign.arch`, runtime prompt or reference
implementation does not establish an observed architecture or implementation stack.
Version fields accept bounded tokens and recognized native compiler banners;
unrecognized banners become unknown, without exposing their raw text.

Unknown facts use conservative canonical wording, such as `Unspecified architecture`;
they do not authorize a specific ISA or version. C/E, the resulting D binding
and its independent reviewer share the frozen context. A new D adopting context
for an older design also retains it for later D operations. Replay restores the
original bytes and intent hash without new discovery or Task interpretation.
Old Task, root identity, configuration, publication, request and review records
are not backfilled.

The existing Light `create_workflow(..., driver_factory=...)` embedding interface
can construct `SchemaWorkflow(..., task_constraint_review=...,
retained_job_evidence=...)`. The two host-side input types currently live in
`aka.workflow.schema._native`; they are not model tools or top-level package exports:

- `TaskConstraintReview(public_task_sha256, constraints)` attests an explicit
  trusted owner review of the **complete** frozen public Task. `constraints`
  contains exactly `root.CONTRACT.Hardware_specification.Architecture`,
  `root.CONTRACT.Hardware_specification.Stack.Language` and
  `root.CONTRACT.Hardware_specification.Stack.Toolchain`, each with
  `status: explicit|unspecified|unknown` and a canonical `value` or `None`.
  `unspecified` requires that complete review; a failed keyword search is
  insufficient. Missing review remains unknown and grants no correction authority.
- `RetainedJobEvidence(raw, raw_sha256, experiment, record_sha256)` supplies the
  original GET bytes and their retained SHA, exact Experiment reference and
  `digest(record.as_dict())` from its native owner. The adapter verifies the
  complete Task, source and shapes, original request, route/namespace, target
  and selected framework. A model attachment or a claimed verification flag
  cannot replace those records. Private payloads stay on the host.

For an exact Planner-owned parent publication, the host can authorize changes
only to those three leaves. Architecture/toolchain factual corrections require
applicable observed evidence; explicit Task constraints must agree. The proof
binds the original parent, operation, context and exact old/new leaf values,
and the host rechecks the entire Contract diff. Model `decision.evidence` grants
no authority. Algorithm, input domain, mathematics, numerical tolerances, ABI,
storage, mutation, Device, other Contract fields and native policy remain frozen.
Imported/extracted designs do not acquire a correction bypass.

A correction publishes a new version with the exact old parent, retaining old
versions, qualification failures and costs. Pure `fact_correction` preserves the
hypothesis, history, working draft, progress, window and consumed budgets; it
does not manufacture a new hypothesis or progress. A genuine execution rewrite
still follows the ordinary new-design gates. New evidence hashes with the same
facts may produce `keep`: D/reviewer retain the parent's original context.
Conflicting established facts require an evidenced new version. Unknown evidence
cannot erase a parent's observed facts. Pipeline changes remain design work for
E, followed by faithful D implementation and new independent qualification.

An old Device display label can be retained only with all three explicit
`TaskConstraintReview` fields: `legacy_device_parent: DesignRef`,
`legacy_device_label: str` and `legacy_device_target: str`. The host verifies the
complete Task, exact target, original label bytes, Planner operation and parent
lineage back to that reviewed anchor. Descendants must preserve the label bytes.
The context's canonical Device remains the actual native target; this grants
neither Device edits nor permission to change GPUs. A new C design cannot use
the historical declaration, and there is no general device-alias allowance.

## Continuous execution and recovery

Planner receives immutable copies of the champion, last tested candidate and
working draft, with exact file hashes, plus bounded native correctness and
measurement summaries. Failed and unpromoted trials remain visible. The host
keeps raw evaluator payloads private and caps source inputs at 256 files/16 MiB
and main feedback at 8,000 characters. Each Planner operation freezes these
inputs before its first call; recovery reuses them. Planner's concrete next
actions accompany the frozen design in the next Coder Episode.

A new design continues from that operation's frozen working draft, even when
the draft failed qualification or was qualified without champion promotion.
With no working draft it uses the frozen champion/baseline. The new hypothesis
and its first D Episode use the same seed; the global champion remains the
separate promotion and performance baseline. This transfers no qualification or
measurement to the new Schema: D still needs new native tests and independent
qualification. `keep` and pure fact corrections retain their existing behavior.
The choice is recorded only for new Planner operations. Older frozen operations
retain their original champion seed rule, and published hypotheses and pending
D bindings are never rewritten on resume.

Design and optimization writers first read the full public Task, hardware,
feedback and source identities, then write an intermediate starter or current
design closure. They consult specifications by relevant field and use candidate
`schema.check` diagnostics while completing the draft. Unfilled starter fields
are allowed only in intermediate output: incomplete drafts do not count as a
completed design, a new hypothesis or promotion. Final contract, numerical,
hardware, closure and publication checks remain unchanged.

The optional Planner setting `"writer_thinking_budget_tokens": 8192` requests a
fixed per-response thinking budget for design/optimize sessions on the Claude
backend. It defaults to `null`, accepts integers from 1024 through 16384, and
requires a concrete model consistent across the native environment and relevant
settings. Bounded settings inspection retains only model/thinking metadata and
file hashes, and repeats after private Home preparation. Conflicting budgets,
disabled thinking, primary `agent` selections and unprovable aliases or redirects
are rejected before invocation. Ordinary permissions, hooks and settings remain
in effect; this does not emulate the CLI's full settings precedence.

The writer session overrides only the selected model's `adaptive_thinking` and
`effort` capabilities, so the pinned CLI sends `thinking.type=enabled` with the
budget and omits `output_config.effort`. Other roles and the default `null` mode
keep their settings; output limits, timeout and root allowance are unchanged.
The earlier budget-plus-effort request was rejected by the real provider.
The revised request has since been accepted in the recorded Qwen run and used
tools, followed by a completed GEMM design; that observation is not a general
guarantee of design quality or completion. Failures do not authorize a silent
model or thinking-mode change. See the [validation record](schema-validation.md).

The `fixed_budget_without_effort` mode, budget, concrete model selection,
settings-file hashes and merged inline settings digest are bound to the writer
operation; its native observation directory also retains `writer-thinking.json`
with the mode and nonsecret identity. The override is in the actual admitted
command. Changed configuration or model/settings sources cannot replay
an old intent; settle its original owner and use
the existing explicit succession/recovery procedure. Existing intents are never
rewritten to adopt the new setting.

Set `continuous: true` and native `max_iters: 0` for continued operation without
a fixed total window/Episode limit. An explicitly supplied positive native
budget remains effective. Optional `limits` fields include `max_episodes`,
`max_windows`, `max_agent_calls`, `max_dispatches`, `max_agent_active_seconds`
and `max_tokens`; for example, `"limits": {"max_episodes": 4}` stops after four
settled Episodes even in continuous mode. Finite mode uses an explicit root
limit, or the positive native `max_iters` when no root total is supplied.
Calls, jobs, windows and repair/retry sequences are individually bounded.
Token receipts may be exact, partial or unavailable; they do not constitute an
absolute provider billing limit.

Each design window contains at most four settled native Episodes. Stagnation
returns work to Planner for a new testable design. The controller retains the
working draft, sealed/tested source and champion separately. Old measurements,
unchanged source, partial measurements and unknown qualification do not count
as new design progress. Independent qualification is a model judgment, not a
formal proof; the native correctness and promotion gates still apply.

Coder's D completion target is faithful implementation of the fixed design.
Once Coder judges it implemented and has a passing standard full Eval for the
exact current source, it retains that source and submits the existing
`candidate_ready` report for independent qualification, even when slower than
the champion or when comparison is unknown. Eval does not establish Schema
fidelity; qualification judges fidelity, and native ABBA/policy gates decide
global champion promotion. Incomplete implementation, concrete conflicts,
bounded exhaustion or infrastructure blockers still permit an evidence-backed
`pivot`/`blocked`. The existing report `summary` carries the
`schema_implementation` mapping, deviations, limitations and original records.
Progress feedback distinguishes a missing retained qualification result
(`qualification-receipt-missing`, which does not establish whether review was
attempted) from `qualification-failed`,
`qualification-unknown` and `no-substantive-change`; these remain `no-progress`.

New D/E requests carry bounded concrete non-pass obligation IDs, reasons and
verified citations from retained qualification. Selection checks the same Task,
design, actual sealed source and execution context. D receives only a review of
its exact current seed; E labels earlier reviewed source as
`historical-reviewed-source`, without presenting it as the current seed.
The view includes original Episode/operation, receipt and binding digests, source
identity, whole-file hashes and readable inline excerpts from the permitted
Task, design and reviewed source. Reasons are reviewer evidence to assess.

The qualification summary is capped at 5,600 characters, 24 non-pass obligations
and eight citations per obligation; E feedback remains capped at 8,000 characters.
Omissions and unavailable citations are explicit. Only the existing four-Episode
window is searched, and a later unreviewed row does not hide an earlier matching
review. Older rows without a summary can be projected from their original native
archive and verified sources for a new request; old rows and frozen requests are
not rewritten. A mismatched or missing receipt supplies no repair authority,
and missing does not establish either an unrequested review or a failed review.

Qualification can correct a completed review's malformed output up to Planner's
`max_repairs` limit (default one). Each correction uses a fresh, admitted and
metered native session with the same frozen Task, hardware, design, candidate and
initial seed, plus the preceding output and protocol diagnostics. Original
attempts remain retained. A valid `fail` or `unknown` ends the review without a
correction; exhausted format errors remain unqualified and unknown. Failed or
unsettled native calls do not receive output corrections. Recovery reuses retained
receipts and terminal results without repeating calls or refunding usage.

Progress uses original terminal Experiment evidence for the independently
qualified, sealed source. Complete current Task correctness and the first full
measurement are finite milestones, including native assessments retained on
attempts. Full coverage requires the frozen reference inputs and all Task Shapes;
native ABBA also requires its complete original schedule. Correctness-only Eval
does not count as performance. A same-record ABBA can establish improvement over
the retained best source for this design without portable environment identity.
Comparisons between records still require matching known environment and
measurement identities, and must beat the design's historical best; promotion
over the global champion is a separate native decision.
The next Coder binding includes the exact retained kernel ID and native
`kernel-read` / `--baseline-path` commands. The shared Campaign source reader
can retrieve measured candidates that were never promoted.

Missing cross-record identity remains unknown. Such a window returns the missing
evidence and a concrete comparison target to Planner, without clearing or adding
measured stagnation. Repeated source or record evidence cannot renew progress;
an unknown window requires an untried next action, and two unresolved windows
require a new hypothesis. Recovery does not rewrite earlier progress, native
measurements or consumed budgets.

Retain the entire private workspace and its sibling `.aka-schema-inputs`
directory. The controller records state beneath
`<workspace>/.atrex_long_horizon/schema`; do not delete state to obtain a new
allowance. Resume in place using the same selection, configuration and request.
Changed task, device, materials or budget identities are rejected. Completed
runs replay their existing terminal result and do not acquire a fresh budget.
Native checkpoint export/import is not supported for Schema workspaces.

Reviewer prompt changes update the shipped release identity and new operation
intent. They do not migrate existing operations: an existing intent rejects a
different prompt, and completed qualification results remain terminal. The
recorded native launch also binds package code and materials. Installing changed
code or prompts is not authorization to replace that capture or silently resume
an older root under the new release; use a separately identified root to validate
the new release when the original identities no longer match.

Unknown model calls or GPU jobs must first be reconciled with their original
owners. The controller does not resubmit them to keep the process moving.
Bounded bad drafts or repeated infrastructure failures leave a diagnostic
`RecoveryRequired` state. After resolving its specific cause, an embedding
supervisor may call `SchemaWorkflow.resume_after_repair(reason)` on the same
controller; this records the repair without refunding consumed allowance.
For a diagnosed preparation-admission defect in an earlier implementation,
the explicit `preparation_campaign=campaign` argument first verifies the same
native context and retained original receipts. Only a provable unaccounted
execution is recorded once as `historical-unadmitted` consumption; it is not
relabelled as an original admission. Ambiguous evidence remains blocked.

Resolve a stopped action before invoking that repair method. It permits a new
bounded attempt after a diagnosed bad draft or infrastructure repair; it does
not settle an unknown model process or GPU job. `recover_agent` must return
evidence from the original owner before the controller can clear an unknown
model call. The public CLI resumes ordinary retained work, but does not provide
an operator command for this explicit repair acknowledgement.

For the diagnosed Legacy Dev timeout/admission failures recorded by older
versions, the original host owner can explicitly call
`LegacyExperiment.settle_unsubmitted(ref, attempt_id, receipt=..., evidence=...)`.
It requires the exact old record, stopped original process identities and
audited pre-submission evidence, and preserves the original record and proof as
private artifacts. Missing JobRefs alone are insufficient. After the original
Experiment is settled, `SchemaWorkflow.settle_unsubmitted_execution(...)` can
settle its matching existing permit without refunding dispatch consumption.
This narrow repair is unavailable to Agent tools and does not resubmit work.
Current managed defaults respect the selected backend's Dev timeout limit;
explicit incompatible deadlines are rejected before dispatch. Typed evaluation
and the Schema root limits retain their separate budgets.

Legacy can recover a transient inline Eval submission inside the same live
call when `idempotent_replay=true` and an explicit gateway URL have been
configured for a service verified to deduplicate identical keyed requests.
This option defaults to false. The current path requires frozen AK/SK
authentication, without token or named/routed profiles; it is tested with
gateway SDK 0.14.36 and needs revalidation when that SDK changes.

The original key, request body, route and job timeout stay fixed. At most two
extra native launches use the existing 5/10-second retry schedule within 120
seconds and the original client deadline; each consumes root/window allowance.
SDK internal HTTP retries are not separate native launches. Original failures
remain private artifacts, and only the same original job's terminal receipt
settles the linked unknown submissions. Once a job ID is known, recovery only
queries or cancels that job. Explicit no-replay policy, corrupted receipts,
missing identity, another unknown owner or a process restart cannot obtain
this live-call permission. Generic scoped replay remains disabled.

Use a separate workspace for a finite diagnostic such as four Episodes; a
finished diagnostic is intentionally terminal. Start the continuous campaign
only after its configuration and observed behavior are accepted, retaining a
link to the diagnostic evidence. A diagnostic does not extend the allowance or
rewrite the task history of an existing campaign.

## Native benchmark inputs

Use `reference.py`, `input.py`, `shapes.json` and measured `metadata.json` beneath
a compatible Atrex-Bench checkout with real `scripts/run_eval.py` and
`src/atrex_bench/`. Shape IDs must parse as integers. The reference is a baseline,
and its metadata must cover the selected hardware and exact inputs.

Select private runtime dependency pins using the native Problem plugin's
`worker_requirements` path. Task preparation freezes this host-only file;
that role takes precedence over `ATREX_SANDBOX_REQUIREMENTS_FILE` defaults.
The public L20D recipe generates the corresponding patch. Keep signed wheel
locators outside the source repository and verify actual installed runtime
identity from the execution receipt.

`EvaluationConfiguration.environment` supplies local host settings. Worker
environment defaults to empty; declare worker values through
`worker_environment` or the existing tool `--env` option. Tool values override
those defaults. Transport-reserved inputs are rejected locally, and the
service validates allowed worker variables. Host credentials and paths are
not inherited by the worker.

To establish new reference metadata with the configured native Agate CLI:

```sh
agate eval --candidate /absolute/problem/reference.py \
  --reference-dir /absolute/problem --gpu L20D --shape-shards 1 \
  --bench-iters 30 --job-timeout 600 --no-wait
agate get ORIGINAL_JOB_ID --wait > /absolute/private/reference-baseline-job.json
```

Query the original job to settlement. After successful correctness and timing,
record each measured microsecond latency at
`metadata.json → shapes[ID].production_performance.performance_us`, set
`perf_gpu`, and retain the original receipt and source/input identities. The
field name does not make a reference a production kernel. The historical
[native vector-add example](../skills/atrex-aka-light/examples/native-atrex-bench-vector-add/README.md)
shows the metadata format and its measurement limits. Fresh native V0 still
runs when a new Schema campaign starts.

## Verification boundary

CPU fixtures cover public assembly, both initial branches, native Episode state,
finite budgets, recovery and installed assets without an upstream checkout.
They use synthetic model output and measurements. Real Claude/Qwen and L20D
campaigns, useful repeated design/implementation progress and a 72-hour
observation window require separate run evidence; these are not established by
the CPU checks or the historical examples.
See the [validation record](schema-validation.md) for current candidate, release
and live-run status.
