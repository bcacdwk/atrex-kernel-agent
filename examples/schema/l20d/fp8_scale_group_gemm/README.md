# No-SOTA grouped GEMM task input

`benchmark/` contains the public reference, input generator, ten shape cases,
roofline and metadata from AKA_experiment
`assets/bench/data/fp8_scale_group_gemm/`, inspected at repository commit
`d709f1eabd5cdcd8347673037583a8810537038c`. Exact copied-byte hashes in `ORIGIN.json`
are authoritative even if an upstream worktree differs from that commit.
No implementation or Schema is provided for this scenario.

The interface is
`Model().forward(a, b, a_scale, b_scale, m_indptr, scale_granularity_mnk=(1,128,128), scale_major_mode='MN', mma_sm=1, out=None, out_dtype=None, backend='trtllm', segment_alignment=1)`.
The reference computes the grouped result, copies it into caller-owned `out`,
and returns that same output. `input.py::_make_inputs` supplies the actual
payloads/scales and stress cases. The ten shape IDs remain numeric strings.
The full valid domain and output mutation contract are in the original
metadata; these ten cases do not prove full-domain correctness.

`published/metadata.json` is unchanged. It contains a source-bound historical
L20D eager-reference calibration, dated 2026-11-20, with the exact reference
SHA-256 `304a60ad870b5282b0d1431c95b769b079ccc2249bddce371db2c024af7a2c78`.
It also retains old pending-preparation fields; no new readiness claim is
inferred from those fields or the earlier experiment.

For AKA's native baseline metadata reader, `benchmark/metadata.json` adds
`perf_gpu=L20D` and projects the ten existing
`performance_baseline.latency_us_by_shape` numbers into
`shapes[ID].production_performance.performance_us`. No latency value is changed
and no new measurement is claimed. `aka_input_preparation` records this exact
adapter and the fresh-V0 requirement. All other original fields are retained.

The baseline is the eager correctness reference, not SOTA. In particular,
the separately located upstream Blackwell umma Schema is not paired with a
L20D promoted kernel by this package and is not included. Start with native
Task preparation, independent acceptance and fresh V0, then design → implement.
The actual chosen evaluator and environment still require the new campaign's
normal validation.
