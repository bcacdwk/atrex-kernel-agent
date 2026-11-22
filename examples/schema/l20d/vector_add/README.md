# Existing L20D kernel and Schema input

`kernel.py` is the unchanged published Triton candidate. The published report
binds it to `published/root.yaml`, the unchanged earlier proposal, on NVIDIA
L20D. `published/results.json` records a 2026-11-21 independent host ABBA ratio
of 1.503522275674508 over the PyTorch reference for three shapes. This is a
historical observation, not new validation. Its environment identity is null;
it does not establish cross-record comparability, statistical significance or
performance over a wider domain.

The public interface is `Model(alpha=1.0).forward(x, residual)`. Both inputs are
equal-length contiguous CUDA bfloat16 vectors; the output has the same shape
and dtype. The cases are `(shape ID, length, alpha)`:
`("0", 1024, 1.0)`, `("1", 65536, 0.5)`, `("2", 1048576, 2.0)`.
The kernel casts alpha to bfloat16, rounds the product to bfloat16, then rounds
the sum to bfloat16. It uses one program per 1024 elements and masks the tail.

| Published artifact | SHA-256 |
| --- | --- |
| `kernel.py` | `105d89d7432e0662625b9388df68ac1f36ecfcecb1bd4494074b3adc7a5f7e5f` |
| `published/root.yaml` | `58a83e9be44e4d1d5f656454fbec1de65ec948f49c79aad06f9daeb6b6d30e1c` |

The source is KernelSchema commit
`dc6e2b3d318f0324ffdb578c3dcc6cf33a64c7e8`,
`examples/workflows/vector_add/`. The final historical measurement identified
KernelSchema `13efd1554d112d56e4d767ba43b66257b2d75669` and AKA
`55bc39f3239e2d9fa65d2a799182a26df413368c`. The reference/input/shape/metadata
files come from AKA's `skills/atrex-aka-light/examples/native-atrex-bench-vector-add/`.
All exact copy hashes and source paths are in `ORIGIN.json`.

## Current-language migration

The published Schema predates the current static language requirements and
does not pass them unchanged. `schema/root.yaml` is an explicit adaptation;
its hash differs from the published file and it has not been measured. The
adaptation preserves the complete algorithm Compute and Math_reshape text,
all four pipeline actions and their descriptions, the 1024 element partition,
worker count, output shapes/dtypes, and all original realization choices.
It makes only these representation changes:

- Bind implementation/reference source strings to the bundled exact hashes.
- Rename the single tolerance key from `output` to the declared output `y`;
  preserve its unknown threshold text without introducing a comparator.
- Register `program` as the existing abstract Triton execution unit and express
  the same entire-program role as `program[0]`; no warp count is invented.
- Use `Refill: none` and `Preassigned: {start: worker_id, count: 1}` to express
  the existing thesis's static program-ID assignment of exactly one tile;
  name completion of that sole assigned tile `Exit: assigned_done`.
- Express the independent loads as `async_unordered`, followed by serial
  compute/store, and translate the original data dependencies into explicit
  Prepared edges. This permits load overlap without inventing async GPU
  instructions or measured concurrency.
- Wrap every original realization string unchanged in `Choice`, with an
  existing upstream `Item` and migration rationale; add the required single-root
  dispatch index.

The complete adapted directory passes `check_design` with
`CheckPolicy(skip_hardware=True, style="advisory")`. Hardware unknowns and style
diagnostics remain visible. The original proposal's statements about earlier
unmeasured tuning remain historical context; they are not updated claims.

The public report does not supply the exact historical comparator threshold.
The new native Task's independent acceptance must be frozen and checked by the
normal evaluator. Neither the historical zero maximum error nor this source
migration justifies inventing a tolerance or a fresh correctness pass.
