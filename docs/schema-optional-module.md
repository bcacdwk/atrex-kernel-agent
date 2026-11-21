# Optional Schema module

For normal use, follow [Schema assistance in native AKA](schema-standard-workflow.md):
`--schema --schema-path FILE_OR_DIRECTORY` retains the original Light tools and
Agate evaluation. It accepts raw design files without the fixed selections or
qualification steps below. The remainder of this page documents the explicit
legacy `--schema-config` integration and its historical CPU evidence.

Schema is off by default. The original `aka workflow` / `aka optimize` parser
accepts `--schema` and `--no-schema`. Only `--schema` imports the optional module,
reads its selection and verifies KS. A configured or installed optional package
does not enable it. With `--no-schema`, even an absent `--schema-config` path is
ignored. Original Workflow DTOs, defaults, prompts, native budgets and exports
are unchanged. A reconstructed launch locks its original selection and cannot
change the switch or configuration.

This module supplies installation, selection, role context and the native
adapter surface. The default factory
`aka.workflow.schema.backend:create_workflow` now implements the
[A5 fixed backend](schema-fixed-backend.md), using A4's exact-source lifecycle
and native audits. Missing transport qualification produces an explicit blocked
result. The explicit [dynamic selection](schema-dynamic-workflow.md) adds a
finite design/fixed/one-optimize/followup chain through that same factory.

## Source installation and provenance

Use Python >=3.11 (the CPU verification used an isolated Python 3.13). Keep the
matching original Framework, Legacy and Light distributions installed as in
[the existing installation guide](application-plugin-migration.md). From an
arbitrary cwd, with `AKA` set to the absolute source checkout:

```sh
git -C "$AKA" -c http.version=HTTP/1.1 submodule update --init -- optional/KernelSchema
python -m pip install -e "$AKA/optional/KernelSchema"
python -m pip install -e "$AKA/packages/atrex-aka-schema"
python -I -c 'from aka.workflow.schema.release import verify_release; print(verify_release())'
```

Only this named submodule is initialized. `.gitmodules` uses the public
`https://github.com/bcacdwk/KernelSchema.git` URL; the gitlink is a detached fixed
commit, not a moving branch. The optional package owns only
`aka.workflow.schema`; its `src` symlink supports editable installation without
copying AKA. The existing `tools/build_distributions.py` also recognizes
`atrex-aka-schema` as an owner. It does not put KS into an AKA wheel: KS uses its
existing source/editable resource contract.

`release.json` records the exact selected KS commit, public URL, AKA A2 base
commit and file hashes/modes. `verify_release()` validates actual imported KS
code and its resource root, independently of cwd, against that explicit list.
It neither scans for a checkout nor downloads anything. An exact materialized
copy of the selected KS checkout also works. The selected manifest covers core,
runtime, Template/extract code, rules, public profiles, the row-softmax resources
and the shared CPU fixture; private runtime qualification captures are excluded.
It does not treat a profile filename, source path or hash as permission to read
that file from a model role.

After an authorized fixed KS update, run
`python "$AKA/tools/freeze_schema_release.py" --commit <full-detached-SHA>`.
The tool refuses another HEAD, a branch or modified selected tracked files.
A7 records the final AKA commit/release pair after implementation; the manifest
does not attempt to contain its own commit hash.

## On selection

`--schema-config` is a JSON object. Its relative paths resolve against the config
directory, not cwd. `request.json` is the original KS implement request with its
verifiable bundle. The profile freezes `parameters.claude` and the same explicit
requested model; no model, Task, Source, profile or factory is inferred.

```json
{
  "format": "aka.schema.selection.v1",
  "request": "bundle/request.json",
  "run_root": "private/native-run",
  "backend": {
    "factory": "aka.workflow.schema.backend:create_workflow",
    "implementation_sha256": "sha256:<actual installed backend.py hash>",
    "parameters": {}
  }
}
```

`factory` may be omitted to select the name above. Its source is resolved and
hashed before importing the factory or its parent packages. The original
config hash is retained in the Bootstrap composition and rechecked on reopen.
The factory's own fixed parameter/qualification policy remains with A5.

```sh
aka optimize --repo-root "$AKA" --problem /absolute/bundle \
  --platform H20 --framework triton --framework-baseline never \
  --convert-after 0 --schema \
  --schema-config /absolute/selection.json

aka optimize --repo-root "$AKA" --problem /absolute/ordinary-problem \
  --platform H20 --framework triton --no-schema
```

The on transformation uses the native Light composition and replaces its
Workflow, Agent and Problem slots. The Agent slot creates the original Agent
owner with a restricted adapter; raw default Claude commands are unavailable.
The Problem slot selects A2's `SchemaProblemProvider`. Default Wiki/plugin
resource bundles are empty in this composition. The remaining common native
modules keep their original owners. Only selected KS Task/acceptance, fixed R,
rules, role Materials and explicitly granted current C/feedback are delivered.
The first profile requires the plain native Light Agent. Existing sandbox or
custom Agent selections are rejected explicitly, so enabling this profile
cannot silently remove a caller's selected process isolation.

Native transport qualification uses the [source delivery probe](schema-source-release.md#capture-the-native-transport).
Its new-scope ceiling defaults to 120000 ms, narrowed to a shorter declared
profile; `--scope-timeout-ms` can explicitly select a positive safe integer up
to that original profile's timeout. The run identity and transaction binding
freeze it. `--continue` accepts no override and keeps the original ceiling,
request and account deadlines. The original profile/hash, USD threshold and
call allowances remain authoritative.

## Concrete A4/A5/A6 interfaces

`SchemaSelection.load(path)` verifies the installed release and fixed request
(or the initial design request of an explicit dynamic selection).
It exposes `document`, `identity`, `source_sha256`, `provenance`,
`request_document()`, `problem_provider(role="coder")`, `claude_profile()` and
`backend_factory()`. It allocates no KS run/account and submits no operation.

`factory(selection, modules)` returns the retained Workflow object; use the
existing Light object/startup contract, including its `_load_problem` and
`_campaign` implementation methods. `NativeModules` contains the original
Agent, Source, CandidateWorkspace, Executor, ExecutionContext,
EvaluatorFactory, evaluation defaults, ExperimentFactory, measurement
parallelism, observer and episode evidence reader/interpreter. Its
`light(agent=..., problem=..., executor=..., evaluator_factory=..., options=...)`
constructs the original `LightWorkflow`. Explicit replacements reach both real
Evaluator and CommandEvaluator through unchanged `ManagedRuntimeFactory.open`.

A5 supplies A2's existing `SchemaEvaluatorFactory` operation resolver and
`SchemaExecution`; it must persist the original operation/locator/reservation
before preparation and gate each actual model, evaluation, retry and audit.
Native ABBA/audits are still independent callbacks and are not qualified by
factory replacement. A5 uses KS's sole `OwnerAccount`, its original authority,
group reservations and receipt absorption. A3 creates no balance, clock or
usage ledger. Selected qualification evidence is an explicit A5 backend/profile
parameter or Ref; A3 never searches private capture directories or interprets
the presence of the K3 public profile as AKA/GPU qualification.

`schema.authority.NativeCallAuthority.bind(run, owner, original_reservation,
source=..., episode=...)` verifies a prior KS `reserve`/`reserve_group` member
against the context's existing authority and `owner.operations`. It does not
reserve, create an account, absorb usage or issue a qualification. Its
`record_locator` writes the existing RunContext intent, including the actual
source/role/request binding; `authorize` invokes that same `OwnerAccount`.
`RoleSession.from_authority(..., authority=..., workspace=..., session_id=...,
output_directory=...)` connects these concrete callbacks. It also uses the
fixed K3 `runtime._local.remaining_timeout_ms` helper to narrow the original
Agent timeout to the reservation's original absolute cutoffs and recheck them
at launch. This internal helper dependency is covered by the fixed KS manifest;
no deadline implementation is copied into AKA. A5 still owns during-work pause,
process recovery and receipt absorption.

`RoleContext.prepare(run_context, role, source=..., source_roles=...,
deliveries=...)` calls K3's actual `prepare_role_input`. Task/acceptance, R/rules,
scope and role authority are fixed. Source changes do not change
`assignment_sha256`. Current C and extra `RoleDelivery` objects have explicit
role grants. Original evaluator bytes are rejected even if renamed. Every
validation reconstructs the exact input from those grants; mutating a payload
while retaining its old assignment hash fails. `retain(directory)` and
`RoleContext.reopen(run, directory, expected_sha256=...)` preserve the exact
payload and grant identity.

`with role.tools(original_agent, empty_scope) as grant:` uses the real scoped
ToolHost. The `schema` binding accepts only `{"op":"read","id":...}`,
`{"op":"effective","node":...}` and `{"op":"check"}`. Read uses granted
material IDs, not filesystem paths. Effective/check use KS's actual Template
API; a nonroot document contains root Contract. Check establishes structure
only. Unknown IDs, evaluator paths, extra path options and all writes fail at
the handler. Tokens are revoked when the tool session closes. These are
**host tools for context preparation/inspection**. The current no-tools model
cannot interactively call them.

`native_agent(ClaudeProfile(...))` returns an original native Agent with only
the restricted adapter. `RoleSession(agent, role, profile, workspace=...,
session_id=..., output_directory=..., authorize=..., record_locator=...)`
uses its real `open_session` and `NativeSession`. `record_locator` is called
before any invocation; A5 must durably bind the locator to its original
operation/reservation. `authorize()` runs in the original Agent's launch hook,
immediately before native execution. A denial starts zero processes. The hook
cannot be bypassed by calling the Agent directly without a registered role.
The request's fixed active limit and native profile timeout bound the native
call. The low-level callback constructor is suitable for synthetic interfaces;
A5 uses `from_authority` for the actual reservation/deadline and must enforce
owner pause, group/fence state and clock during work. A canonical original
session UUID and explicit current Source are mandatory.

`session.invoke(old_prompt, purpose="initial"|"repair"|"rate_limit_retry"|
"resume"|"reopen")` always supplies the fixed role input, replacing the old
free-exploration prompt. One object performs at most one call. The profile has
no hidden context/native resume: `resume=True` and a second invocation reject.
Reopen/repair/retry means a new owner-authorized operation from explicit stable
C after settling the old one. A5 must fence lost native returns/process state;
A3 does not implement K3's detached process recovery as a second runtime.

The tiny `stdin_exec` shim runs under the original Agent's process owner,
verifies retained input and executable bytes, binds stdin, and `exec`s the CLI
in the same PID. It avoids the per-argument Linux limit for complete Rules.
It has no retry, subprocess, credential discovery or result transformation.

`NativeRoleResult` retains the full `NativeInvocation`, raw stdout/stderr,
normalized usage/events and observation errors. It verifies actual native
permission/model/session initialization and parses `RoleOutput` against the
fixed assignment. An invalid permission/output result keeps its original usage
but supplies no candidate. `native_source()` returns the exact original
`RevisionContent({"kernel.py": utf8_bytes})`, preserving CRLF/non-ASCII/missing
final newline. The host can pass it directly to
`GitCandidateWorkspace.install_source(draft, result.native_source())`.
`freeze_source(publication, candidate_workspace)` durably publishes exact C and
returns A2's `SourceBinding` (separate KS manifest/native/kernel identities).

A4 owns initial C0 and seed intent/ready/admitted recovery at A1's preparation
seam. A5 owns subsequent authorized draft installation, original
`seal_candidate`/evaluation calls, receipt settlement and result publication.
Neither result conversion method installs C0, seals, evaluates or promotes.
The explicit dynamic workflow consumes these fixed role/material interfaces and
calls A5's one implementation. The role/session adapters themselves start no
planner or outer loop.

A4's concrete [source lifecycle](schema-seed-lifecycle.md) now supplies both the
seed transaction and an `install_result(e, result, authority=...)` helper for A5
to accept and persist one authorized A3 result. Sealing, evaluations, fees and
audits remain with A5.

## Scope of validation

The CPU tests use actual original Agent/NativeSession/ToolHost,
ProblemProvider/CandidateWorkspace, installed metadata, KS publication and
Template. The model process and fee authorization are explicitly synthetic.
Tests cover default/off isolation and equality, on provider selection, host
role read/write negatives, nonroot Contract, fixed assignment on first/reopen
and explicit changed C, original complete usage retention, exact candidate
roundtrip and a real fresh child through stdin exec. Existing A1/A2 tests retain
their own scope.

The first native profile is `aka.schema.claude.no-tools.v1`: implement
coder/reviewer, single file, no empty start, no interactive model tools, native
context resume or hard token enforcement. Finite token budgets and zero role
allowances are denied. The trusted CLI's explicit flags enforce the model tool
boundary; this is not an OS sandbox for a malicious CLI binary. K3's public
Claude profile/flags are reused where applicable, but its runtime qualification
does not transfer to this AKA transport. Actual AKA native/model/GPU
qualification require their own real records. A5's three audit and direct paths
are implemented in the [fixed backend](schema-fixed-backend.md); the bounded
native capture and source delivery procedure are documented in
[source delivery](schema-source-release.md).
