# Schema workflow

Install matching `atrex-aka-schema`, `atrex-aka-light`, `atrex-aka-legacy` and
`atrex-aka-framework` distributions. Schema owns design, extract, optimize and the
fixed-design implementation driver. Planner designs and revises Schema (C/E); Coder implements
each frozen design (D) in the normal Light Episode engine. The selected native Agent and
Executor own model calls, correctness, measurement and promotion. Design fidelity and champion
promotion remain separate gates.

KernelSchema is not a runtime, build or submodule dependency. The migrated language,
specification, starter and selected Wiki are installed package data. Fixed source hashes and
attribution remain in [SOURCES.json](../src/aka/workflow/schema/assets/SOURCES.json) and
[Schema NOTICE](../src/aka/workflow/schema/NOTICE.md); the upstream snapshot had no root
license declaration, and AKA does not claim to grant a new license for those portions.

Build all four distributions from the same source revision:

```sh
python -m pip install build 'packaging>=23' 'PyYAML>=6'
python tools/build_distributions.py --output /absolute/fresh-distributions \
  atrex-aka-framework atrex-aka-legacy atrex-aka-light atrex-aka-schema
python -m pip install --no-deps --force-reinstall /absolute/fresh-distributions/*.whl
```

Use a dedicated environment with the authenticated Agent CLI and native execution
prerequisites. Version `0.2.0` alone does not identify a release; retain its commit and wheel
hashes for each real run.

## Start

Use the installed CLI with an explicit Schema configuration:

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

`selection` is the resolved AKA profile with its Agent, Executor and private environment.
Schema uses that native model CLI and configuration. A prebuilt `WorkflowLaunch` must
explicitly set `framework_baseline="never"`, `convert_after=0`, no baseline reviewers and
`setup_timeout` within the Schema per-call limit; continuous operation also needs
`max_iters=0`. New CLI requests receive these defaults when omitted. See the [L20D launch
examples](../examples/schema/l20d/README.md).

The configuration has `format: "aka.schema.launch.v1"`. Its three entry branches are:

- `initial`: a verified same-target kernel and complete Schema pair. Exact source
  hashes and retained evidence are checked against native preparation before E;
  the native `source` plugin selects those kernel bytes.
- Neither `initial` nor `sota`: fresh native V0 preparation, then design C.
- Explicit source-only `sota`: six-stage extraction. A reference implementation
  does not implicitly select this branch; the L20D examples do not use extraction.

Relative Schema/evidence paths resolve from the configuration file. Supply the whole design: `root.yaml`,
`dispatch.md` and every referenced node. The vector-add example's adapted historical proposal
uses `skip_hardware: true`; this diagnostic does not replace independent target identity,
native correctness or current GPU measurement. Omitting `--schema` preserves normal Light
behavior without loading Schema's parser or assets. A workspace retains its original mode;
switching requires a new workspace.

Schema preserves native `resources`, including bundles and explicit plugins/catalogs; Coder
receives them through the Light workspace installer. Planner's selected Schema documents,
starter and Wiki have a separate identity: do not assign those materials to
`workflow.resources`. The L20D recipe selects `measurement-session` and `research-records`
without an external Wiki checkout. Older material-only resource locks are incompatible and are
not migrated or rewritten.

## Frozen execution facts and Contract corrections

Execution context binds the prepared and complete public Task, target hardware and native
policy hashes; the selected framework comes from that policy. Architecture and toolchain facts
come from applicable preparation observations, an owner-verified retained Job or the selected
gateway's GET-only environment discovery. For fresh preparation, a complete successful Eval's
original owned Job takes precedence over its snapshot after Task, prepared-source, request,
route and namespace checks. Missing or unqualified receipts retain the snapshot source. SSH
reuses preparation observations without another remote probe. Facts retain their evidence
hashes and `observed`, `configured`, `declared`, `cache` or `unknown` provenance.

A GPU name, `campaign.arch`, prompt or reference implementation does not establish
architecture or stack. Version fields accept bounded tokens and recognized compiler banners;
unknown values remain conservative (for example, `Unspecified architecture`). C/E, D and the
independent reviewer share the frozen context. A new D adopting context for an older design
retains it for later D operations. Replay restores original bytes and intent without
discovery, reinterpretation or backfilling old records.

The Light `create_workflow(..., driver_factory=...)` interface can construct
`SchemaWorkflow(..., task_constraint_review=..., retained_job_evidence=...)`. These host-only
types live in `aka.workflow.schema._native`, not model tools or top-level exports:

- `TaskConstraintReview(public_task_sha256, constraints)` attests trusted review of
  the **complete** public Task. The exact three keys are
  `root.CONTRACT.Hardware_specification.Architecture`,
  `root.CONTRACT.Hardware_specification.Stack.Language` and
  `root.CONTRACT.Hardware_specification.Stack.Toolchain`; each has
  `status: explicit|unspecified|unknown` and a canonical `value` or `None`.
  `unspecified` needs complete review; a keyword search or absent review grants
  no correction authority.
- `RetainedJobEvidence(raw, raw_sha256, experiment, record_sha256)` supplies original
  GET bytes, retained SHA, exact native Experiment and `digest(record.as_dict())`.
  The adapter verifies Task, source, shapes, request, route/namespace, target and
  framework. Model attachments or verification claims cannot replace owner records;
  private payloads stay on the host.

For an exact Planner-owned parent, only those three leaves can be corrected.
Architecture/toolchain corrections require applicable observed evidence and agreement with
explicit Task constraints. Host proof binds parent, operation, context and exact old/new
values; the host checks the entire Contract diff. Model `decision.evidence` grants no
authority. Device, all other Contract fields and native policy stay frozen; imported/extracted
designs gain no bypass. Each correction publishes a new version with its exact parent,
preserving old versions, failures and costs. Pure `fact_correction` preserves hypothesis,
history, working draft, progress, window and consumed budgets. Execution rewrites follow
ordinary new-design gates and fresh qualification. Same facts with new evidence hashes may
produce `keep`, retaining the parent's context for D/reviewer; conflicting established facts
need an evidenced new version, and unknown evidence cannot erase observed facts.

A historical Device display label requires all three additional review fields:
`legacy_device_parent: DesignRef`, `legacy_device_label: str` and `legacy_device_target: str`.
The host verifies the complete Task, exact target, original label bytes, Planner operation and
lineage to that anchor; descendants keep the label bytes. Canonical Device stays the actual
native target. This permits no Device edit, GPU change, general alias or historical
declaration in a new C design.

## Continuous execution and recovery

Set `continuous: true` and native `max_iters: 0` for operation without a fixed total
window/Episode limit. Explicit positive native budgets still apply. Optional `limits` are
`max_episodes`, `max_windows`, `max_agent_calls`, `max_dispatches`, `max_agent_active_seconds`
and `max_tokens`; `"limits": {"max_episodes": 4}` stops after four settled Episodes. Finite
mode uses an explicit root total or positive native `max_iters`. Calls, jobs and repair/retry
sequences are individually bounded. Token receipts may be exact, partial or unavailable, not
an absolute billing limit.

Each design window contains at most four settled native Episodes; stagnation returns work to
E. The controller retains working draft, sealed/tested source and champion separately. Planner
freezes immutable source copies and hashes plus bounded native feedback before its first call;
recovery reuses them. Source inputs are capped at 256 files/16 MiB and feedback at 8,000
characters; raw evaluator payloads stay private. A new design and its first D share that
operation's frozen working draft, including failed or qualified-but-unpromoted drafts; without
one they use champion/baseline. The champion remains the promotion/performance baseline. No
qualification or measurement transfers to the new Schema. Older frozen operations keep their
original seed rule; resume does not rewrite published hypotheses or pending D bindings.

Writers first read Task, hardware, feedback and source identities, then write an intermediate
starter/current closure, consulting relevant specifications and `schema.check`. Incomplete
intermediate drafts earn no design completion, hypothesis or promotion. Final Contract,
numerical, hardware, closure and publication gates apply.

Optional `"writer_thinking_budget_tokens": 8192` requests fixed per-response thinking for
design/optimize writers on the Claude backend. Default is `null`; valid budgets are integers
1024–16384. A concrete model must agree across native environment and relevant settings, and
any configured output limit must exceed the budget. Bounded settings inspection currently
requires Linux, retains model/thinking metadata and hashes, and repeats after private Home
preparation. Conflicting budgets, disabled thinking, primary `agent` selections and unprovable
aliases/redirects are rejected before invocation; protected managed thinking/effort selections
cannot be silently overridden. Only the selected model's `adaptive_thinking` and `effort`
capabilities are overridden for this writer: the admitted command sends
`thinking.type=enabled` with the budget and omits `output_config.effort`. Other roles and
default `null` keep their settings. Permissions, hooks, output limits, timeouts and root
allowance remain in effect; this does not emulate the CLI's full settings precedence or permit
silent model/thinking-mode changes. The `fixed_budget_without_effort` mode, budget, model, settings-file
hashes and inline settings digest bind the intent; `writer-thinking.json` retains the
nonsecret identity. Changed model/configuration/settings cannot replay an old intent: settle
its original owner and use explicit recovery/succession, without rewriting the existing
intent.

D targets faithful implementation. Once Coder judges the fixed design implemented and has a
passing standard full Eval for the exact current source, it retains that source and reports
`candidate_ready` for independent qualification, even if slower than champion or comparison is
unknown. Eval establishes correctness, qualification
judges fidelity, and native ABBA/policy gates decide promotion. Qualification is model
judgment, not formal proof. Incomplete work, conflicts, bounded exhaustion or infrastructure
blockers permit evidenced `pivot`/`blocked`. Report `summary` carries the
`schema_implementation` mapping, deviations, limitations and original records.
`qualification-receipt-missing`, `qualification-failed`, `qualification-unknown` and
`no-substantive-change` are no-progress; a missing receipt does not establish whether review
was attempted.

D/E feedback includes verified non-pass obligation IDs, reasons and citations bound to the
same Task, design, sealed source and execution context. D sees only its exact seed's review; E
labels earlier sources `historical-reviewed-source`. Views retain Episode/operation,
receipt/binding digests, source hashes and permitted inline excerpts. Limits are 5,600
characters, 24 non-pass obligations and eight citations per obligation; omissions are
explicit. Search covers only the current four-Episode window; a later unreviewed row cannot
hide a matching review. Older rows can be projected from verified native archives into new
requests without rewriting old rows or frozen requests. Missing/mismatched receipts grant no
repair authority. Malformed completed reviews allow up to Planner `max_repairs` (default one)
fresh, admitted, metered correction sessions with the same frozen inputs and prior
diagnostics. Valid `fail`/`unknown` end review; exhausted format errors remain
unqualified/unknown. Failed or unsettled calls get no output correction. Recovery never
repeats retained terminal reviews or refunds usage.

Progress requires original terminal Experiment evidence for the qualified sealed source. Full
correctness and first full measurement are finite milestones; full coverage needs frozen
reference inputs, every Task Shape and the complete ABBA schedule where applicable.
Correctness-only Eval is no performance result. Same-record ABBA may prove design improvement
without portable environment identity; cross-record comparison requires known matching
environment/measurement identities and improvement over that design's historical best. Global
champion promotion is separate. D receives retained kernel IDs and
`kernel-read`/`--baseline-path` commands, including measured unpromoted candidates. Repeated
source/record evidence cannot renew progress. Missing comparison identity remains unknown and
returns a concrete evidence target without changing measured stagnation; an unknown window
needs an untried next action, and two unresolved windows require a new hypothesis.

Retain the private workspace and sibling `.aka-schema-inputs`; state is under
`<workspace>/.atrex_long_horizon/schema`. Normal resume uses the same selection, configuration
and request. Task, device, materials or budget changes are rejected; completed runs replay
their terminal result without fresh allowance. Schema does not support native checkpoint
export/import. Package code, materials and reviewer prompts bind launch/operation identities:
changed installations do not authorize replacing captures or silently resuming old roots. Use
a separately identified root when those identities no longer match.

For `RecoveryRequired`, resolve the diagnosed cause and stopped action, then call
`SchemaWorkflow.resume_after_repair(reason)` on the same controller for a bounded attempt
without refund. The public CLI resumes ordinary work but has no explicit
repair-acknowledgement command. Unknown model calls/jobs require original-owner
reconciliation; `recover_agent` must return original-owner evidence. Repair does not itself
settle unknown work or authorize resubmission.

| Recovery entry | Applicable evidence and effect |
| --- | --- |
| `resume_after_repair(reason, preparation_campaign=campaign)` | Diagnosed historical preparation-admission defect; verifies the same native context and original receipts. Records only provable unaccounted execution once as `historical-unadmitted`, without inventing original admission. Ambiguity stays blocked. |
| `LegacyExperiment.settle_unsubmitted(ref, attempt_id, receipt=..., evidence=...)`, then `SchemaWorkflow.settle_unsubmitted_execution(...)` | Original host owner of a diagnosed Legacy Dev timeout/admission failure; exact old record, stopped original processes and audited pre-submission evidence are required. Missing JobRefs are insufficient. Retains the original record, proof and charges, settles only the matching permit, never resubmits; unavailable to Agent tools. |
| Legacy `idempotent_replay=true` | Opt-in recovery of transient inline Eval/Profile/Check/Disassemble submission inside the **same live call**, with an explicit gateway verified to deduplicate identical keyed requests. Requires frozen AK/SK, no token/named/routed profile; tested SDK is 0.14.36 and changes need revalidation. |
| `LegacyExperiment.reconcile_cancelled_eval(ref, timeout=30)`; `resume_after_repair(reason, campaign=original_campaign)` | Original-owner confirmation of an already cancelled inline Eval after Coder timeout/archive; also checked before normal continuation. One owned Job GET may settle logical record/permits as terminal/inconclusive, never measure, qualify or promote. |

Keyed live recovery retains the original key, complete body, route and namespace before
submission. At most two extra admitted launches follow 5/10-second delays within 120 seconds
and the original deadline; each consumes allowance. Auxiliary operations make one POST per
admitted launch; SDK transport/overload retries cannot bypass it. Eval keeps its native SDK
HTTP retries. Only the same Job's terminal receipt settles linked unknown submissions. Known
Job IDs permit only query/cancel; no-replay, corrupt/missing identity, another unknown owner
or process restart cannot gain permission. Generic scoped replay stays disabled; historical
unknowns cannot acquire a new key. Profile sends `options.timeout_s`; Check/Disassemble retain
local budgets and report `job-runtime-budget-unconfirmed`, which establishes no remote
deadline and masks no HTTP failure. Ordinary recorded auxiliary calls preserve caller keys on
the wire. Managed Dev defaults respect backend timeout limits; incompatible explicit deadlines
are rejected before dispatch. Typed evaluation and root limits keep separate budgets.

Cancelled Eval confirmation requires the original lock, input/key, source, physical receipts,
live association and namespace. It queries the original Job once with `include=spec` on its
original SDK connection, redirects/retries off, no artifacts; the read is allowed after
measurement budget expiry. Cancelled status, completed cleanup and non-quarantined execution
must be explicit, without conflicting termination/cleanup evidence. Optional source/request
echoes may be absent/null; returned comparable fields must agree. Original records, physical
unknowns and consumed dispatches/Coder time stay archived; the query has no physical attempt
ID. Missing/conflicting evidence blocks.

Use a separate workspace for finite diagnostics. A finished diagnostic is terminal; continuous
campaigns retain a link to its evidence without extending old allowances or rewriting old task
history.

## Native benchmark inputs

Use `reference.py`, `input.py`, `shapes.json` and measured `metadata.json` in a compatible
Atrex-Bench checkout with real `scripts/run_eval.py` and `src/atrex_bench/`. Shape IDs must
parse as integers; reference metadata must cover hardware and exact inputs. Select private
runtime pins with the native Problem plugin's `worker_requirements` path, frozen during
preparation and taking precedence over `ATREX_SANDBOX_REQUIREMENTS_FILE`. The L20D recipe
generates this patch. Keep signed wheel locators outside the repository; verify installed
identity from execution receipts. `EvaluationConfiguration.environment` sets local host
values. Worker environment is empty by default; use `worker_environment` or overriding tool
`--env` values. Transport-reserved values are rejected locally and the service validates
worker variables; host credentials and paths are not inherited.

Establish reference metadata with the configured native Agate CLI:

```sh
agate eval --candidate /absolute/problem/reference.py \
  --reference-dir /absolute/problem --gpu L20D --shape-shards 1 \
  --bench-iters 30 --job-timeout 600 --no-wait
agate get ORIGINAL_JOB_ID --wait > /absolute/private/reference-baseline-job.json
```

Query the original Job to settlement. After successful correctness and timing, set
`metadata.json → shapes[ID].production_performance.performance_us` to measured microseconds,
set `perf_gpu` and retain receipts and source/input identities. The field name does not make a
reference a production kernel. The [native vector-add
example](../skills/atrex-aka-light/examples/native-atrex-bench-vector-add/README.md) documents
the format and measurement limits. A new campaign still performs fresh V0.

## Verification boundary

CPU fixtures cover public assembly, entry branches, Episode state, budgets, recovery and
installed assets with synthetic model output and measurements. Independent installation
establishes the packaged release, not model/GPU behavior. Real model/L20D execution, repeated
useful C/E→D progress and 72-hour stability each need run evidence. See the [validation
record](schema-validation.md) for release and run status.
