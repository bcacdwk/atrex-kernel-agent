# Schema workflow

Install matching `atrex-aka-schema`, `atrex-aka-light`, `atrex-aka-legacy` and
`atrex-aka-framework` distributions. Schema owns design, extract, optimize and
the fixed-design implementation driver. The selected native Agent and Executor
still own model calls, correctness, measurement and promotion. Planner writes a
complete design; Coder implements it in the normal Light Episode engine.

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

## Continuous execution and recovery

Planner receives immutable copies of the champion, last tested candidate and
working draft, with exact file hashes, plus bounded native correctness and
measurement summaries. Failed and unpromoted trials remain visible. The host
keeps raw evaluator payloads private and caps source inputs at 256 files/16 MiB
and main feedback at 8,000 characters. Each Planner operation freezes these
inputs before its first call; recovery reuses them. Planner's concrete next
actions accompany the frozen design in the next Coder Episode.

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
