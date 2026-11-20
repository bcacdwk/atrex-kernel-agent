# Finite dynamic Schema workflow

The original `aka workflow` and `aka optimize` commands select this workflow
only with `--schema` and an explicit `backend.parameters.dynamic` object.
The default factory remains `aka.workflow.schema.backend:create_workflow`.
An empty parameters object still selects the [fixed backend](schema-fixed-backend.md),
which never calls a planner. Schema remains off by default.

This workflow supports one row-softmax full Task, `single_file/kernel.py`, and
a finite sequence: optional initial design, one fixed implementation stage,
at most one optimize, and at most one requested fixed followup. It calls the
installed KS `api.design` / `api.optimize`, then executes each fixed stage with
the existing native `FixedController`. It does not call the KS demo driver or
start another outer implementation loop.

The engineering tests use actual KS APIs and native Campaign/Store/Git objects,
with explicitly synthetic model and evaluation transports. No actual A6 model
or GPU qualification has been completed. Unconfigured native services return
the original zero-fee blocked result. The earlier native coder timeout and
unknown usage are not qualification for this workflow.

## Existing Schema or extract output

Prepare an ordinary immutable KS implement request selecting `backend.name=aka`,
the exact Schema/Scope, and an explicit Source or `seed=null`. This supports the
same three consumption choices as fixed execution: verify the exact existing
source with `verify_only`, continue from an explicit source, or rebuild from the
Task starter with `seed=null`. The request's `budget.parent` must reference an
already-created KS `OwnerAccount` descriptor. The existing private authority is
an explicit path; the workflow never creates a replacement parent balance.

The optimize template is a hash-pinned original KS design or optimize request.
Its Task and parent must match. Its Rules, profile, Materials and authorized
`workflow_objective` remain fixed; the actual optimize request uses the current
Schema/Scope and adds the previous fixed stage's settled evidence. A design
request is accepted as a template so preparing the finite chain never requires
inventing a future Schema. `optimize_request: null` disables optimize.

```json
{
  "format": "aka.schema.selection.v1",
  "request": "bundle/implement-request.json",
  "run_root": "private/dynamic-run",
  "backend": {
    "factory": "aka.workflow.schema.backend:create_workflow",
    "implementation_sha256": "sha256:<installed backend.py digest>",
    "parameters": {
      "dynamic": {
        "start": "schema",
        "authority": "private/owner.json",
        "implementation": null,
        "optimize_request": {
          "path": "bundle/optimize-template.json",
          "sha256": "sha256:<exact request-file digest>"
        }
      }
    }
  }
}
```

Paths resolve relative to the selection file. Ref objects inside requests and
the implementation descriptor retain their original bundle-relative paths.
All selected closures must exist locally; no input or credential discovery
takes place. Obtain a file digest with `sha256sum`, prefixing the result with
`sha256:`. Native transport and evaluation qualification still follow the
[source release procedure](schema-source-release.md).

```sh
aka optimize --repo-root "$AKA" --problem /absolute/bundle \
  --workspace-base /absolute/private/native \
  --platform H20 --framework triton --framework-baseline never \
  --convert-after 0 --token-budget 0 --schema \
  --schema-config /absolute/selection.json
```

`aka workflow` accepts the same selection. Repeating the same command resumes
the original run. Use the original selection, native settings, workspace,
profile, installed implementation and fixed KS release.

## Starting with no Schema

Set `request` to a valid KS **design request** and `start` to `design`. No Schema
is required in this request, and none is supplied to the planner. Set
`implementation` to the following explicit descriptor, using Refs from that
request's bundle:

```json
{
  "profile": {"path": "artifacts/<id>/aka-profile.json", "sha256": "sha256:<digest>"},
  "rules": {"path": "artifacts/<id>/rules.json", "sha256": "sha256:<digest>"},
  "materials": {"path": "artifacts/<id>/materials.json", "sha256": "sha256:<digest>"},
  "budget": {
    "parent": {"path": "artifacts/<id>/parent.json", "sha256": "sha256:<digest>"},
    "limits": {
      "calls": {"planner": 0, "coder": 4, "reviewer": 8, "extract": 0},
      "evaluator_jobs": 8, "active_ms": 1200000, "tokens": null
    }
  },
  "mode": "code",
  "seed": null
}
```

The descriptor contains no Schema. After KS completes and reviews its design,
the workflow forms a new immutable implement request from those inputs and the
published Schema/Scope. The implementation Rules must include the actual
`fidelity.freeze_method`; writing Rules/profile come from the existing
`writing.freeze_rules`, `writing.backend_profile` and `writing.request` helpers.
The Task, parent, native profile and seed choice never change implicitly.

Both writing profiles must freeze `parameters.writing_finish_floor` using KS's
complete `budget.cost` vector. For the default leaderboard native settings:

```python
from kernelschema.budget import cost

fixed = cost(role="coder", calls=1, evaluator_jobs=2, active_ms=4000)
review = cost(role="reviewer", calls=1)
writing_profile["parameters"]["writing_finish_floor"] = {
    key: fixed[key] + review[key] for key in fixed
}
```

Publish that profile before constructing `writing.request`. Its request limits,
parent limits and any native limits must also allow the reserved calls/jobs/time.
KS's planner reservation adds its independent reviewer minimum to this floor;
the writing reviewer retains the floor as well. These reservations reserve
capacity without charging nonexistent work. Production delivery and configured
V0 prefetch reviewers require a larger floor, computed from the native profile's
explicit `minimum_active_ms`; insufficient configuration is rejected before the
dynamic chain starts. The workflow does not reserve a fictitious extra operation
or create another account to keep this capacity.

## Stage policy and recovery

Multiple coder episodes remain inside the same fixed R. Ordinary build,
numerical or fidelity failure/unknown, budget exhaustion and lack of native
promotion do not trigger optimize. Only a completed fixed stage with an
explicit authorized optimize objective proceeds. This version has no trusted
independent design-blocker protocol and does not infer blockers.

An optimize result with no action stops. An explicit `implement` action selects
exactly one Source or Task `default_start`; the latter produces `seed=null`.
An explicit `verify` action selects exactly one nonempty Source and executes
`verify_only`, with zero coder calls and no new native episode. A revised R is
published before the next fixed request is prepared and admitted. New R has no
current qualified C until its own fixed stage supplies one.
Additional action input references beyond Schema/Scope and the Source or Task
marker are explicitly unsupported by this finite adapter. Their writing output
is retained with a blocked/new-request result; they are never silently discarded
or granted to a new role.

The workflow retains one native repository and writer lease. A code-stage
handoff checks the old fixed finish, every original operation, closed episode,
unchanged Task, exact HEAD/checkpoint, source state and native limits. It reuses
the settled V0 and its reviewer records, retains cumulative episode/version,
stall and promotion history, and begins a new source lifecycle at the next
native episode. The exact selected seed replaces the new draft's source. No
existing draft, immutable request, old scope or old qualification is rebound.
The new Task snapshot is captured only at this explicit new-Q boundary, before
any intent. Losing an active stage's snapshot does not permit reconstruction.

Private state records request intent/ready, account binding before episode creation,
source ready/admitted, original result selection, old scope closure and writing
publication separately. `finishing` resumes its original paid work first.
Missing outer `result.json` is recoverable from the retained child run and
receipts. Unknown original calls/jobs retain their run/owner/channel and block
new stages. An existing child run missing its budget binding fails closed;
absence of a record never creates another account or grants a fresh allowance.
`new_request_bound` means the original account has bound the new immutable Q;
it does not claim fee admission. Each actual fixed operation still reserves and
authorizes against the original owner pause, fence and remaining limits.

`<run_root>/dynamic-result.json` is a stage summary, with the immutable child
request/result references and native HEAD history. Each child `result.json`
remains the original KS result. `current_qualified` belongs to the summary's
current Schema. `native_result.schema` identifies the request that produced
that native result; its incumbent G is separately reported and is not a claim
that G passed the current Schema. Previous qualified C and old R remain in
history. All children reference the same parent authority and cumulative clock;
usage and receipt deduplication remain with KS `OwnerAccount`.
