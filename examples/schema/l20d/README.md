# Public L20D experiment inputs

These inputs prepare the two Schema workflow scenarios without invoking extract.
They contain public source and compact published provenance, with no private
runtime configuration or raw experiment logs. The actual Agent/Executor and
independent target Environment must be supplied by the normal AKA launcher.
The reference experiment configuration requests `qwen3.8-max[1m]` through
Claude Code 2.1.316 with bwrap/projected and the native LegacyExecutor. The
provider previously reported `qwen3.8-max`; the 1M capacity is not independently
established here.

| Scenario | Task input | Starting candidate | Starting design |
| --- | --- | --- | --- |
| Existing measured kernel and supplied Schema | `vector_add/benchmark/` | `vector_add/kernel.py` | `vector_add/schema/` |
| No SOTA; design then implement | `fp8_scale_group_gemm/benchmark/` | None; reference is only the correctness/performance baseline | None |

Each `ORIGIN.json` binds the copied upstream file hashes. Source strings in the
adapted vector-add Schema use `aka-public@<file-sha256>:<local-relative-path>`;
they identify these bundled files, with paths relative to `vector_add/`.
There is no runtime dependency on KernelSchema or AKA_experiment.

Use a real compatible Atrex-Bench checkout containing `scripts/run_eval.py`
and `src/atrex_bench/`. Copy the selected `benchmark/` directory into a new
`data/<problem-name>/` directory there; these inputs do not include an evaluator
runtime. Both directories pass AKA's CPU shape and reference-baseline structure
checks. No additional wrapper is needed: both expose `Model.forward` and
`_make_inputs`. Loading structure does not import or execute either GPU source.

The supplied `schema-config.json` files select the new AKA controller. For
vector-add, the selected native source plugin supplies the exact kernel bytes;
the controller publishes the complete adapted Schema, binds the prepared Task
and computes the native source identity after fresh V0. `source_files` contains
raw input hashes, not a substitute for that native identity. It enters optimize
directly. The candidate is a historically measured starting point, not a claim
of globally optimal SOTA.

For the no-SOTA scenario, do not provide an initial Schema or SOTA descriptor.
Native preparation freezes the new Task/reference/acceptance and performs its
fresh V0, then the workflow calls design. Existing reference timings are
historical input metadata, not results from the new campaign. Full correctness,
current target GPU identity, measurement and promotion must run again for both
scenarios after the implementation/commit sequence.

See [the vector-add migration](vector_add/README.md) and
[the grouped GEMM input notes](fp8_scale_group_gemm/README.md) for the specific
interfaces, changes and remaining evidence limits.

## Launch through AKA

Install matching AKA Framework, Legacy, Light and Schema wheels in the dedicated
run environment. No KernelSchema package or checkout is required. Copy this
whole public input directory to a stable location if desired. Set these paths
to your installed inputs, compatible Atrex-Bench checkout and new external run
directory:

```sh
export AKA_INPUTS=/absolute/public-inputs/l20d
export AKA_BENCH=/absolute/atrex-bench
export AKA_RUNS=/absolute/private-runs/l20d-schema
umask 077
mkdir -p "$AKA_RUNS/config"
test ! -e "$AKA_BENCH/data/aka_schema_vector_add"
cp -R "$AKA_INPUTS/vector_add/benchmark" "$AKA_BENCH/data/aka_schema_vector_add"
test ! -e "$AKA_BENCH/data/aka_schema_fp8_scale_group_gemm"
cp -R "$AKA_INPUTS/fp8_scale_group_gemm/benchmark" "$AKA_BENCH/data/aka_schema_fp8_scale_group_gemm"
```

Load the established private environment for the native Legacy transport and
Claude authentication. It must set `PATH` to the qualified Claude and bwrap
executables and provide the native GPU route and evaluator requirement. Keep
that environment file and its credentials outside this repository; do not copy
the experiment repository's launcher or configuration wholesale.

Use AKA's `ATREX_SANDBOX_REQUIREMENTS_FILE` for native runtime dependency pins
and `ATREX_AGATE_EXECUTABLE` for the selected Agate executable. Keep signed wheel
locators in that private requirements file. Also select that file as the native
Problem's `worker_requirements`, as generated below: its frozen Task role takes
precedence over environment defaults. The experiment wrapper's
`ATREX_SANDBOX_ATREX_BENCH_REQUIREMENT` variable is not an AKA configuration
interface. Select an optional evaluator version only when the actual gateway
supports it for the chosen operation; verify the installed evaluator identity
from the original receipt.

```sh
. /absolute/private/l20d-runtime.env
export ATREX_SANDBOX_REQUIREMENTS_FILE=/absolute/private/runtime-requirements.txt
export ANTHROPIC_MODEL='qwen3.8-max[1m]'
export ANTHROPIC_DEFAULT_OPUS_MODEL='qwen3.8-max[1m]'
export ANTHROPIC_DEFAULT_SONNET_MODEL='qwen3.8-max[1m]'
export ANTHROPIC_DEFAULT_HAIKU_MODEL='qwen3.8-max[1m]'
export ATREX_CLAUDE_SESSION_SETTINGS='{"skipWebFetchPreflight":true}'
```

Freeze the selected private runtime pins, and create the explicit native source
selection for the paired scenario:

```sh
python - <<'PY'
import json, os
from pathlib import Path
inputs = Path(os.environ['AKA_INPUTS']).resolve()
runtime = Path(os.environ['ATREX_SANDBOX_REQUIREMENTS_FILE']).resolve(strict=True)
worker_patch = Path(os.environ['AKA_RUNS']) / 'config/worker-requirements.patch.json'
worker_patch.write_text(json.dumps({'api_version': 1, 'patches': [
    {'id': 'problem', 'config': {'worker_requirements': str(runtime)}}]}, indent=2) + '\n')
destination = Path(os.environ['AKA_RUNS']) / 'config/vector-source.patch.json'
destination.write_text(json.dumps({'api_version': 1, 'patches': [
    {'id': 'source', 'config': {'kind': 'kernel',
     'kernel_demo': str(inputs / 'vector_add/kernel.py')}}]}, indent=2) + '\n')
PY
```

Start the **existing pair** with this command:

```sh
aka workflow --repo-root "$AKA_INPUTS" --profile light-sandbox \
  --patch "$AKA_INPUTS/runtime.patch.json" \
  --patch "$AKA_RUNS/config/worker-requirements.patch.json" \
  --patch "$AKA_RUNS/config/vector-source.patch.json" \
  --schema --schema-config "$AKA_INPUTS/vector_add/schema-config.json" \
  --problem "$AKA_BENCH/data/aka_schema_vector_add" \
  --workspace-base "$AKA_RUNS/vector" --workspace-suffix l20d_schema \
  --platform cuda --framework triton --agent-cli claude --hardware L20D \
  --optimization-mode production --expose-evaluator-shapes \
  --framework-baseline never --convert-after 0 --fast-episodes 0 \
  --max-iters 0 --max-stall 0 --target-util 0 --token-budget 0 \
  --setup-timeout 3600 --production-review-timeout 900 \
  --timeout 3600 --verify-repeats 2 --verify-run-timeout 600 --shape-batch-size 10
```

Start the **no-SOTA grouped GEMM** with this command in its own run directory:

```sh
aka workflow --repo-root "$AKA_INPUTS" --profile light-sandbox \
  --patch "$AKA_INPUTS/runtime.patch.json" \
  --patch "$AKA_RUNS/config/worker-requirements.patch.json" \
  --schema --schema-config "$AKA_INPUTS/fp8_scale_group_gemm/schema-config.json" \
  --problem "$AKA_BENCH/data/aka_schema_fp8_scale_group_gemm" \
  --workspace-base "$AKA_RUNS/gemm" --workspace-suffix l20d_schema \
  --platform cuda --framework triton --agent-cli claude --hardware L20D \
  --optimization-mode production --expose-evaluator-shapes \
  --framework-baseline never --convert-after 0 --fast-episodes 0 \
  --max-iters 0 --max-stall 0 --target-util 0 --token-budget 0 \
  --setup-timeout 3600 --production-review-timeout 900 \
  --timeout 3600 --verify-repeats 2 --verify-run-timeout 600 --shape-batch-size 10
```

Both configurations are continuous with bounded individual calls/jobs. An
operator can add `--schema-limits '{"max_episodes":4}'` for a deliberately finite
run; preserve that exact configuration on resume. The native request uses
`framework_baseline=never` to avoid a separate V1/model baseline; fresh native
V0 still runs. Runtime/episode timeouts are 3600 seconds, queue timeout is 60,
request timeout is 7500, and each native ABBA verification uses two repetitions
with a 600-second per-run timeout. These follow the inspected experiment
parameters without inheriting its former 100-Episode total limit.

`runtime.patch.json` sets measurement parallelism to one within each process.
For two concurrent chains, retain the established allocation isolation: distinct
physical allocations, or the existing shared measurement admission/serialization
rule for the same GPU. A value of one in each process is not a cross-process GPU
lock. Query and settle any retained job before resuming; never start a second
copy of the same run to bypass a pending owner.

The complete source/Schema provenance and historical benchmark numbers are
unchanged by these launch files. These commands are experiment preparation, not
evidence that real model/GPU runs or multiple days of stability have completed.
See [the workflow guide](../../../docs/schema-workflow.md) for state, budget and
recovery semantics.
