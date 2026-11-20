# Schema Task adapter (A2)

This optional code consumes KS commit
`1a6c20171e2e02caf1c793738aeb9ab7ed2b6df6` and AKA A1
`a668f026ee30eab9cfc9832ecd1a0d110244adc6`. It does not enable a plugin/switch,
install KS, implement seed transactions, or grant a GPU/profile qualification.
The changes outside `aka.workflow.schema` are explicit adapter-ID branches in
`aka.evaluation.tool_requests.request_from_argv`, Light's Task admission and its
V0 harness contribution. The Schema Task uses its injected evaluator without a
fake Atrex-Bench directory or public test_kernel.py. Ordinary Task paths do not
import Schema or KS.

## Task and source

`SchemaProblemProvider(task, scope, materials=None, role="coder")` implements the
existing `ProblemProvider` API. `load(bundle_root, snapshot_directory=private)`
returns a real `LoadedProblem` and uses the original AKA role snapshot codec.
Snapshot restore validates the selected Task/scope/materials/role and exact
bytes/modes, without needing the original input directory.

| Original role | Schema mapping |
| --- | --- |
| `AgentProblem.files` | Exact Contract, acceptance, and explicitly selected role Materials. Private evaluator/input aliases are rejected, including renamed identical bytes. |
| `LoadedProblem.reference/public_contract` | Public `contract.yaml`; it is not a Python reference wrapper. |
| `InitialSource(kind="kernel")` | Verified `Task.baseline_source`, the exact executable FP32 softmax plus copy baseline. Existing `LegacySourceProvider` consumes it directly. |
| `EvaluationInput(adapter="kernelschema-row-softmax-v1")` | Private canonical bundle closure plus Task/scope binding. No evaluator files are copied through `runtime_files()`. |

The original `private_reference_dir` field is a legacy private-shapes sentinel;
it stays `None`. Privacy comes from explicit EvaluationInput roles and the
host's filesystem policy, not from detecting a directory or `valid.py`.
`content.assert_private()` checks public material integrity and common accidental
bootstrap copies; A3 still owns actual role filesystem allowlists.

`selected_seed(bundle, task, seed)` resolves null to Task.starter_source and
validates it with KS. Invalid explicit seeds never fall back. This function does
not install C0. A4 installs C0 after V0 at A1's preparation seam. `never` is not a
V0 bypass. V0 does not require R0 fidelity: this baseline deliberately uses
external softmax and is not the R0 fused implementation.

`freeze_native_source(publication, workspace, candidate_workspace,
revision=None, expected=None)` returns `SourceBinding`. It reads through the
real native CandidateWorkspace, accepts only complete single-file `kernel.py`
with mode 0644 and strict UTF-8, and preserves CRLF, missing final newline and
non-ASCII bytes. `solution.json` remains metadata; auxiliary declarations and
local auxiliary imports are rejected. The binding records public Source Ref,
native source identity, kernel byte digest, size and mode. For this native
single-file codec, the native identity is SHA-256 of the exact kernel bytes;
the public C identity is the separate canonical Source manifest digest.

## Actual evaluation interfaces

```python
from kernelschema.evaluation import LocalExecutor
from aka.contracts.evaluation import EvaluationConfiguration, EvaluationRequest
from aka.task.candidate_workspace.provider import GitCandidateWorkspace
from aka.experiment.runner import LegacyExperiment
from aka.workflow.schema.evaluation import (
    EvaluationOperation, LocalChannel, SchemaExecution, SchemaEvaluatorFactory,
)

execution = SchemaExecution(private_ops, LocalChannel(LocalExecutor(private_jobs)))
# The sole owner persists this locator and reservation before calling prepare.
locator = execution.locator("v0-1")
operation = EvaluationOperation("v0-1", "original-key", owner,
    {"owner": owner, "evaluator_jobs": 1, "active_ms": 60000},
    expected_source=baseline_ref, mode="v0")
factory = SchemaEvaluatorFactory(execution=execution, operation=operation)
evaluator = factory.create(EvaluationConfiguration(timeout=60),
    candidate_workspace=GitCandidateWorkspace(), evaluation_input=problem.evaluation)
verification = LegacyExperiment(execution, evaluator).run(
    EvaluationRequest(agent_workspace, mode="v0"))
evidence = execution.read_operation("v0-1")
```

`SchemaEvaluator` implements `prepare/observe_step/interpret`; `SchemaExecution`
implements the real native `Executor` contract (`CommandSpec`/`JobResult`).
Preparation freezes source and an immutable intent in the private operation root.
`PreparedEvaluation.inputs` is empty: no evaluator enters the Agent workspace.
KS stages and verifies the full executable closure, then its existing controller
loads candidate/baseline directly from verified original bytes. No AKA wrapper
repairs outputs, calls allclose, or reimplements timing/comparison.

For a long-lived Runtime, the factory's `operation` may instead be a synchronous
`EvaluationRequest -> EvaluationOperation` resolver supplied by A5. It must
return the original owner/key/reservation on re-prepare/recovery; calling prepare
again is not permission to allocate a new operation or balance. The adapter
does not implement admission/accounting inside this callback.

The evaluator requires full Task scope before submission. It runs the original
36 cases/72 basic calls, two-call state reuse, original out storage/view/backing,
complete writes, x invariants, prefix/suffix guards, binary64 golden tolerance,
and all 80 ordered complete-forward timing batches. Construction, first call,
warmups and samples remain in the original report. The public timing soft goal
is independent of Task validity and native incumbent promotion.

`factory.create_command(...)` supplies the actual CommandEvaluator used by native
V0 and full Evaluate. The thin tool-request branch avoids the legacy construction
of reference.py/input.py/shapes.json. The Task owns cases and timing; subset,
explicit timing/seed overrides, arbitrary commands and baseline comparisons fail
before execution. `EvaluationOperation.mode="v0"` binds an unversioned native V0
request; explicit `--version v0` does so as well. Bind expected_source to baseline
for V0. Native output exposes status and public coverage-shape latency only.

CPU success remains `UNAVAILABLE`/`all_pass=false`, with original CPU report
checks preserved as unknown for CUDA. CPU logical failures yield FAIL. A trusted
actual CUDA report with pass build/correctness/performance, matching loaded C,
environment, input bytes and complete samples has a reachable native Task PASS
path (including V0); it does not require current R fidelity. Its native result
has no incumbent or improvement percentage and is not an ABBA result.

`AgateChannel(existing_AgateDevExecutor, profile=..., environment=...,
frozen_inputs=...)` uses the same Task/source/result mapping and the implemented
KS prepare/submit/query/settle API. It consumes an actual prior probe capture;
it never finds credentials, installs remote tooling, probes or changes channels.
The host must admit that channel from actual auth/dev ACL, pinned bwrap/namespaces,
CUDA access, never-replay semantics, output retention and cleanup evidence.
This A2 run did not submit any GPU job. A2.real remains unverified.

## Receipts, reuse and A5 seams

`execution.consume(prepared.context, raw_job_result)` or
`execution.read_operation(operation_id)` re-queries the original KS terminal,
checks original absorption/report closure, actual loaded bytes, Task/evaluator,
input/timing, actual environment, frozen RNG bytes and samples. It returns the
original report Ref/document, private worker facts, bundle, receipt ID and source
binding. Reading an old operation does not reprepare against today's workspace.

`require_applicable(evidence, task_bundle=..., task=..., source_binding=...,
channel=..., environment=..., frozen_inputs_sha256=...)` checks explicitly selected
current requirements and returns the *unchanged old report Ref*. It is not a
generic untrusted-report validator or a Schema/native qualification decision.
Wrong C/evaluator/inputs/environment/timing/profile cannot be fixed by relabeling
an old pass. A5 separately binds current Schema/rules/scope/fidelity.

Actual integration points:

* `light/runtime.py:ManagedRuntimeFactory.open` creates both Evaluator and
  CommandEvaluator. A3/A5 inject the Schema services there through original
  composition. Factory selection alone does not replace Runtime services.
  Light's Task admission now permits this explicit adapter without runtime_root;
  its existing setup skips an absent harness only for this adapter and still
  performs native InitialSource/V0 handling.
* `light/orchestration/campaign.py:_run_v0_evaluator` calls
  `measure_for_acceptance` → `SupervisorRuntime.execute_trusted` → original
  tool request construction → CommandEvaluator. CPU cannot finish real V0.
* `light/long_horizon/recorded_verifier.py:RecordedABBAValidator.verify` uses its
  `execute` callback and `same_allocation_abba` records; it does not automatically
  consume the new Evaluator. `GatewayABBAValidator` also owns an independent
  comparison route. A5 must gate/replace those actual callbacks and each audit,
  using original source revision, C mapping, Task E, current fidelity and owner
  operation. Current Schema command routing explicitly rejects these requests.
* `evaluation/native_abba.py:prepare_comparison` prepares Atrex-Bench reference
  fields and native comparison observations. The current
  `drivers/native_harness.py` forwards to canonical Atrex-Bench run_eval; neither
  path establishes this Task's full storage/guard/golden64 hook. `valid.py` is
  even excluded by the legacy portable gateway collector. No installed local
  Atrex-Bench implementation or service hook was qualified in this session.

The owner still handles all admissions, V0/V1/prefetch costs, active clock,
unknown cross-channel fences and native three-audit/comparison policy. The
adapter tightens the worker reservation to the native timeout but does not own
the cumulative parent balance; preparation/query time belongs to that owner.
Retries use the same original key and KS controller. A lost/corrupt original
intent is an error; waiting cancellation is not proof of remote cleanup.

## Source and qualification record

New files here are AKA adapter code; KS parser/evaluator/comparator code is not
copied into AKA. CPU testing uses a `git archive` of the fixed KS commit, not an
editable install from its changing runtime worktree. The selected archive paths
are pyproject.toml, README.md, kernelschema/, schema_template/, schema_extract/,
schema_docs/, examples/workflows/row_softmax/, the shared row-softmax fixture,
and its evaluation logic/execution tests. Archive SHA-256:
`48c504ce3083c37e0aa97a0b422108f7eb77b16ad3290b077d15400bb7031433`.
The complete selected-file hash manifest is retained at
`/tmp/schema-b04-a2-ks-provenance.json`; A3 owns the final installed release pair.

Tests use real native Task/Source/Experiment objects and the real KS CPU
transport; only the separately labelled CUDA-success projection unit exercise
uses hypothetical facts. No CUDA, Agate service, native ABBA, actual promotion
or whole-plugin integration is claimed by these CPU tests.
