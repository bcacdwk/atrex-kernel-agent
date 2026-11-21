# Fixed Schema source delivery

This source profile supports Linux, an explicit installed AKA source checkout,
the fixed KS source/resource checkout, and the original single-file row-softmax
Task. It implements the fixed controller and the explicit
[finite dynamic workflow](schema-dynamic-workflow.md). It does not support
FA4/source trees, V1 conversion, model tools, native context resume or hard
token budgets. Python 3.13 has been exercised; Python 3.11 compatibility remains
unverified.

| Source | Fixed provenance |
| --- | --- |
| Original AKA | <https://github.com/alibaba/atrex-kernel-agent>, base `bf5557ac04f6ff9988901d511cf25e1cbe61a1bc` |
| AKA integration fork | <https://github.com/bcacdwk/atrex-kernel-agent>, dynamic final-review input `04fab50d67f37f341fa0eb60f8a1928d65df1d10` |
| KS dependency | <https://github.com/bcacdwk/KernelSchema>, detached gitlink `7d5fb60cf1d6f9ef7cf63dff757250bb4f614ab1` |
| KS selected resource manifest | `src/aka/workflow/schema/release.json`, SHA-256 `0ac0141bac462dfaafa07c66bb44d3db1b215c8a12f6d9e73e75eb99a3f34a00` |

The review input above precedes the accompanying final-review changes. A
published release must identify the final reviewed AKA commit and wheel hashes;
`0.2.0` alone is not source identity. Do not resume an admitted run on changed
code. Keep its original wheels, source/resources, private run and parent account
available to settle original work.

The existing AKA [LICENSE](../LICENSE) and [NOTICE](../NOTICE) apply to AKA's
source. KS and other dependencies retain their own licensing and notices; an
AKA wheel does not include or grant a license to those repositories. Private
qualification captures, credentials, author HOME and runtime accounts do not
belong in a public source archive.

The [capability and pairing index](schema-support-evidence.md) separates the
CPU engineering paths, the selected transport qualification, and each backend's
remaining Task and release gates.

## Install and check

Use one environment and a reviewed checkout; initialize only the selected KS
gitlink. The original builder stages source ownership and builds wheels from
sdists, including AKA's existing LICENSE/NOTICE.

Install runtime dependencies before the AKA wheels: the KS editable installation
supplies `PyYAML>=6.0,<7` and `jsonschema>=4,<5`; Framework/Legacy also require
`packaging>=23`. Force reinstall the four AKA wheels because their shared
`0.2.0` version can otherwise leave older bytes installed. `--no-deps` keeps
that step from replacing the selected KS editable installation.

```sh
git -C "$AKA" submodule update --init -- optional/KernelSchema
python "$AKA/tools/build_distributions.py" --output "$DIST" \
  atrex-aka-framework atrex-aka-legacy atrex-aka-light atrex-aka-schema
python -m pip install -e "$AKA/optional/KernelSchema" 'packaging>=23'
python -m pip install --no-deps --force-reinstall "$DIST"/*.whl
python -I -c 'from aka.workflow.schema.release import verify_release; print(verify_release())'
python -I -m aka optimize --help
python -I -m aka.workflow.schema.native_probe --help
```

Keep KS as its supported source/editable resource installation. Importing a
different or modified KS resource tree fails release verification. Installation
does not enable Schema: normal `aka workflow` and `aka optimize` remain off;
`--no-schema` ignores the schema-config path.

## Capture the native transport

Start from an immutable, complete KS **code** request using the intended
`profile.parameters.claude`. The request needs coder/reviewer allowance and
unlimited tokens; its original parent and native active/call limits still apply.
The public KS row-softmax `run_demo.py prepare --start fixed` can prepare such a
request without a model or GPU call. Use its explicitly selected target and
runtime profile when preparing real business work.

If that request has a parent account, supply the existing live authority path.
Do not construct a second owner from its public descriptor:

```sh
export KS_BUDGET_AUTHORITY="$DEMO/.runtime/owner.json"
python -I -m aka.workflow.schema.native_probe \
  --request "$DEMO/fixed.request.json" --out "$PRIVATE/native-qualification"
python -I -m aka.workflow.schema.native_probe \
  --continue "$PRIVATE/native-qualification"
```

This finite command uses AKA's original Agent/NativeSession and RoleCalls under
the same OwnerAccount. It invokes coder once and reviewer once, each bounded by
the declared profile, the frozen qualification scope ceiling and the original
owner deadlines, with a native USD threshold no greater than 0.50. It
never starts an evaluator/GPU job, installs generated code into a business
draft, or claims Task/fidelity success. The result is
`native-qualification-result.json`; `usage` is the original owner receipt
projection and `cli_reported_usd` is CLI reporting, not an independently checked
bill. A failed role stops the probe; continuation does not retry a paid call.

The new scope's default ceiling is **120000 ms**, narrowed to the declared
profile when that profile is shorter. A request whose original
`profile.parameters.claude.timeout_ms` is at least 300000 can explicitly select
a 300-second qualification ceiling at creation:

```sh
python -I -m aka.workflow.schema.native_probe \
  --request "$DEMO/fixed.request.json" --out "$PRIVATE/native-qualification-300s" \
  --scope-timeout-ms 300000
python -I -m aka.workflow.schema.native_probe \
  --continue "$PRIVATE/native-qualification-300s"
```

An explicit value must be a positive safe integer no greater than the declared
profile's timeout. The scope identity and original transaction binding freeze
this value before any call. `--continue` reads that original value, rejects
`--scope-timeout-ms` overrides and never replaces it with the current default.
Missing original scope records or bindings fail closed; retain the original
source and artifacts to settle an older run. There is no migration of admitted
scopes or retry of a failed paid call.

The probe preserves the exact declared Claude profile and its hash. Each call
is still narrowed by the original request, parent and native reservation
deadlines; this option grants no new account, active-time allowance, calls or
USD budget. The qualification artifact records the scope ceiling and verifies
it against the original locator and exec evidence. A larger ceiling proves
only permissions and terminal recovery over the observed execution; it does
not prove an actual run lasting that long. Business RoleSession calls retain
the explicitly frozen profile interval and the original owner's tighter limits.

The two recorded AKA native trials each stopped after one coder timed out at
120 seconds; neither called a reviewer or GPU. The latter used a 355182-byte
input at source `e867333dfcbbce553465635bcf57a9bde842b06f`. Both have unknown
tokens and CLI-reported USD and produced no qualification. Their original
requests, accounts, failed results and source artifacts remain unchanged;
this configurable ceiling adds no real transport qualification.

Successful qualification retains both original NativeInvocations, exact role
inputs, operation/session IDs and pre-exec PID/boot/start-tick records. Admission
rechecks those selected original bytes, executable/profile, actual no-tools
permission init, allowed models, role output and complete token usage. A K3
runtime capture does not qualify this AKA transport. Synthetic injections are
marked and cannot be selected. A fabricated “pass JSON” is not a supported
qualification step; these local captures depend on their trusted original
execution provenance and are not service-signed attestations.

Unreturned native processes remain `settle_only` at their original locators.
The tool does not infer zero usage from missing files or start a replacement.
The core-anchor/budget-binding initialization gap remains a conservative error;
retain it for diagnosis rather than automatically creating another run.

## Select captured evidence and run

`select_request` creates a **new** immutable request/selection from the original
request and actual capture. It copies only selected input/exec/invocation refs
as `roles=[]` Materials. It preserves the original Task/R/scope/rules, exact
seed, K6 evaluation parameters and parent descriptor, and never copies the live
account or edits an admitted request.

```python
from aka.workflow.schema.native_probe import select_request

select_request(
    "/absolute/prepared/qualified-fixed.request.json",
    "/absolute/private/native-qualification",
    "/absolute/prepared/aka-selection",
    run_root="/absolute/private/aka-run",
)
```

The input above must select the existing K6 Agate channel and its original
probe/baseline/four-negative suite to run the full Task. KS's documented
`run_demo.py qualify` and `kernelschema.evaluation.channel.qualify` capture
those six paid jobs using the existing official configuration, explicit target,
GPU alias and installed remote Python/bwrap. Preserve its returned selection
using `kernelschema.evaluation.channel.selection`; do not hand-author successful
worker results. No Agate service/image installation is performed by AKA.
Missing K6 selection or qualification returns blocked.

```sh
aka optimize --repo-root "$AKA" --problem "$SELECTED" \
  --platform H20 --framework triton --framework-baseline never \
  --convert-after 0 --token-budget 0 --schema \
  --schema-config "$SELECTED/selection.json"

kernelschema implement --request "$SELECTED/request.json" --out "$PRIVATE/ks-entry"
kernelschema implement --continue "$PRIVATE/ks-entry"
```

Use one direction for a given run. Both enter the same fixed controller and
original native owner; the KS direction does not nest another implement loop.
The process exit code is distinct from the result's domain `status`.

## Release gates and A6 handoff

CPU engineering checks and native transport qualification are separate gates.
A7.fixed acceptance additionally needs the actual selected GPU/full Task,
independent current-R fidelity, complete production delivery and reconciled
original fees. Actual Agate same-allocation ABBA has not been qualified; the
concrete service leaves comparison unselected. A qualified but unpromoted C can
complete. Different-allocation timing cannot sign native ABBA.

A6 consumes `FixedController.execute()` or `Backend.execute(original_context)`
and the unchanged KS result. Settle and close the old transaction before a new
immutable request for changed R/mode, preserving the original parent authority.
Ordinary code/fidelity failure is same-R work; it does not itself authorize a
planner. Use the original campaign lease; a caller already holding it retains
its controller. Preserve exact C separately from native G/HEAD and commit-bound
qualification. A committed promotion missing qualification retains native
history and stops Schema progression. Paid finish recovery stays in `finishing`
until original production/native receipts have been absorbed.
