# Schema assistance in native AKA

Schema is optional and off by default. Select a raw Schema file or directory to
give its design to the normal Light coding agent:

```bash
aka optimize --repo-root /path/to/atrex-kernel-agent \
  --problem /path/to/atrex-bench/data/operator \
  --platform L20D --framework Triton --agent-cli claude \
  --workspace-base /path/to/runs --max-iters 4 \
  --framework-baseline never --convert-after 0 \
  --schema --schema-path /path/to/schema
```

A directory contributes its YAML nodes and `dispatch.md`. Use repeatable
`--schema-doc FILE` for selected explanations or task constraints. The design is
frozen into the normal launch selection and supplied to each fresh coding
episode. Same-session repairs retain it in their existing conversation.

The selected Agent, tools, ProblemProvider, source layout, Executor, Evaluator,
resources and Light campaign remain the ordinary AKA components. Schema adds
no writing review, semantic/fidelity acceptance, transport qualification, custom
GPU sandbox, probe or fixed case-count requirement. Production mode still uses
the normal AKA production policy when explicitly selected. Existing Light
patches, including `cuda_manifest` and `light-sandbox`, keep their usual meaning.

## KernelSchema calls AKA

The synchronous Python entry accepts the same raw files and an exact campaign
workspace. KernelSchema imports it lazily; neither repository needs a submodule
pointing back to its caller.

```python
from aka.workflow.schema.standard import implement

result = implement(
    schema="/path/to/schema", problem="/path/to/atrex-bench/data/operator",
    repo_root="/path/to/atrex-kernel-agent", workspace="/path/to/run/native",
    platform="L20D", framework="Triton", max_iters=4,
    docs=("/path/to/rationale.md",),
    prepare_only=True,  # Remove to run the native workflow.
)
```

`settings` and `runtime` accept the existing `WorkflowSettings` and
`CampaignRuntimeSettings` fields. `profile` and `patches` select the normal
composition. `prepare_only` reads the design, resolves the composition and loads
the native Task; it does not boot services, call a model or submit a GPU job.
It does not prove a remote environment or a performance baseline is ready.

The Python API preserves the exact `workspace` path through native startup,
including its campaign lease and result collection. The ordinary CLI keeps its
generated `kernel_opt_*` child directory convention. Inherited `BASH_FUNC_*`
shell functions are ignored when preparing native workload environments;
explicit `--env` inputs retain their usual validation.
The native shell guard also retains its helper functions when Claude restores a
Bash snapshot; ordinary Python tool commands and the existing import/path
restrictions continue to use the same wrapper.

## Native Task and performance baseline

Use the original native Atrex-Bench layout: `reference.py`, `input.py` and
`shapes.json` under a real Bench checkout containing `scripts/run_eval.py` and
`src/atrex_bench` source, rather than empty marker files. Shape keys must parse as
integers, for example `"0"`, `"1"`, `"2"`; the worker uses `int(shape_id)` for
input seeds. The three files suffice for Task loading. Normal full evaluation
also requires measured `metadata.json` baselines for every shape on the selected
hardware, including in leaderboard mode.

For a new operator, use the existing configured Agate CLI to measure its reference:

```sh
PROBLEM=/path/to/atrex-bench/data/operator
agate eval --candidate "$PROBLEM/reference.py" --reference-dir "$PROBLEM" \
  --gpu L20D --shape-shards 1 --bench-iters 30 --job-timeout 600 --no-wait
# Use the job ID returned above; query the same job until it finishes.
agate get JOB_ID --wait > /path/to/reference-baseline-job.json
```

After successful correctness and timing results, save each actual shape latency
in microseconds at
`metadata.json → shapes[ID].production_performance.performance_us`.
Set `perf_gpu` to the measured hardware and retain job ID, input identity and
measurement provenance with the original receipt. The historical field name
`production_performance` does not establish that the measured source is a
production or optimized kernel. A PyTorch reference measurement is a reference
baseline; do not insert guessed numbers or apply another device's timings.

The [vector-add example](../skills/atrex-aka-light/examples/native-atrex-bench-vector-add/README.md)
includes the exact JSON format and a recorded L20D reference baseline for its
three unchanged bf16 shapes. Its provenance and remeasurement instructions
explain the limits of reusing those values.

## Results and feedback

The return value is JSON serializable:

| Field | Meaning |
| --- | --- |
| `status`, `exit_code`, `stop_reason` | Native workflow completion/failure and stop reason; completion alone does not assert correctness or promotion |
| `workspace`, `schema_identity` | Exact native workspace and selected design identity |
| `evaluation_scope` | The supplied Atrex-Bench reference/input/shapes and normal native ABBA promotion |
| `versions` | Original version records, including their correctness, measured performance and failure facts |
| `native_state` | Original episode counts, attempts and usage state when present |
| `artifacts` | Existing kernel/manifest, native result, version records and feedback paths |
| `feedback_files` | Original version and episode result files whose text a planner can consume |

Read `completed` or a version's `PASS` together with its `candidate_commit` and
performance `measurement_subject` / `measurement_source`. For example,
`canonical_incumbent_carry_forward` with subject `incumbent` and no candidate
commit reports the previous baseline; it does not establish a new kernel's
correctness, measurement or promotion.

`schema-result.json` retains the complete return value. Repeating the Python call
with the same completed workspace and request returns that retained result;
changing the request requires a new workspace. For interruption, keep the native
workspace and receipts and use AKA's existing recovery. A changed Schema starts
a new implementation stage, and KernelSchema's driver can feed the actual
native results to its next writing step. No old K1/strict Task qualification is
claimed by these standard results.

## Budgets

Without `owner_account`, the existing AKA limits and usage records apply. The
normal `max_iters`, token admission and episode time controls remain unchanged.

For an explicitly shared KernelSchema account, pass `owner_account` as the
existing private authority path and `owner_limits={"model_calls": 2,
"evaluator_jobs": 4}`. An existing active parent request can also supply
`owner_run_id`; otherwise a native request is registered in that same account.
This optional adapter reserves a finite group, counts each native CLI invocation
and each distinct confirmed GPU job, records reported tokens, and releases only
unused slots. Native CLI invocations may contain multiple provider turns; this
is not a count of individual model HTTP requests. Unknown jobs remain unsettled
and cannot silently receive another allocation. Query/cancel retain the original
job; automatic replay is disabled for the shared-account adapter. The adapter
requires the parent's token field to be unbounded because AKA reports native
invocation token totals after execution. Parent active time still bounds work.

The equivalent optional CLI flags are `--schema-owner-account`,
`--schema-owner-limits` and `--schema-owner-run-id`. No account is created by
these flags, and none of them is needed for an ordinary standard run.

## Install and legacy compatibility

Use Python >=3.11 and the normal [runtime prerequisites](quickstart.md#prerequisites),
including coordinator `torch`, an authenticated coding CLI and a configured
evaluation backend. With `AKA` set to this checkout's absolute path, install
its selected KernelSchema source and all four matching AKA packages in the
same Python environment:

```sh
AKA=/path/to/atrex-kernel-agent
git -C "$AKA" -c http.version=HTTP/1.1 submodule update --init -- optional/KernelSchema
python -m pip install build 'packaging>=23'
python -m pip install -e "$AKA/optional/KernelSchema"
AKA_SCHEMA_DIST=$(mktemp -d "${TMPDIR:-/tmp}/aka-schema-dist.XXXXXX")
python "$AKA/tools/build_distributions.py" --output "$AKA_SCHEMA_DIST" \
  atrex-aka-framework atrex-aka-legacy atrex-aka-light atrex-aka-schema
python -m pip install --no-deps --force-reinstall "$AKA_SCHEMA_DIST"/*.whl
aka optimize --repo-root "$AKA" --help
kernelschema implement --help
```

The editable install provides KernelSchema's declared Python dependencies;
`packaging` and all four local AKA wheels satisfy the AKA package dependencies.
`--no-deps` keeps pip from replacing the selected KernelSchema source with an
index package. It does not install the runtime prerequisites above.
`--force-reinstall` replaces earlier packages even though all four still use
version `0.2.0`; that version alone does not identify their source revision.
Keep those packages and the AKA resource checkout at the same commit.

Only the named KernelSchema submodule is initialized. The gitlink and
[`release.json`](../src/aka/workflow/schema/release.json) record its exact source
commit; an independent source/editable checkout at that same commit also works.
This checkout selects
[`KernelSchema@907c99aa1589bddf73e26079f2304b83d68a8e51`](https://github.com/bcacdwk/KernelSchema/tree/907c99aa1589bddf73e26079f2304b83d68a8e51).
The standard raw-file path does not require running legacy release or transport
qualification. [The optional-module guide](schema-optional-module.md#source-installation-and-provenance)
describes the legacy manifest verifier.

`--no-schema` and omission of the switch keep the original launch path and do
not import or inspect the optional Schema module. `--schema-config` explicitly
selects the historical strict controller for old requests; its old qualification
rules and receipts remain separate. It is mutually exclusive with
`--schema-path`.
