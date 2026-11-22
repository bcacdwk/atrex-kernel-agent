# Minimal Native Atrex-Bench input

This directory shows the smallest useful Native Atrex-Bench problem accepted by AKA
Light:

- `reference.py` defines the authoritative `torch.nn.Module` semantics;
- `input.py` defines `_make_inputs`, which turns one Shape's `input_kwargs` into CUDA
  tensors passed to `Model.forward`;
- `shapes.json` maps stable integer-string Shape IDs to `Model.__init__` and
  `_make_inputs` arguments; the Agate worker uses `int(shape_id)` when seeding inputs.

The three files suffice for Task loading. Normal full performance evaluation also
needs measured `metadata.json` for every Shape on the selected hardware, including
in leaderboard mode. This example includes a real L20D reference baseline described
below. `roofline.json`, `valid.py`, and `agent_problem.json` remain optional.

This directory is stored in the Skill, not under a canonical Atrex-Bench runtime. Copy
it into the `data/` tree of the Atrex-Bench checkout that will evaluate it:

```bash
cp -R \
  /path/to/atrex-kernel-agent/skills/atrex-aka-light/examples/native-atrex-bench-vector-add \
  /path/to/atrex-bench/data/native-atrex-bench-vector-add
```

Use the real Atrex-Bench source checkout, with both runtime paths above the operator:

```text
/path/to/atrex-bench/
├── scripts/run_eval.py
├── src/atrex_bench/
└── data/native-atrex-bench-vector-add/
    ├── reference.py
    ├── input.py
    ├── shapes.json
    └── metadata.json
```

Empty marker files are insufficient. With the matching baseline ready, select it with:

```bash
aka optimize --repo-root /path/to/atrex-kernel-agent \
  --problem /path/to/atrex-bench/data/native-atrex-bench-vector-add \
  --platform L20D \
  --framework Triton \
  --agent-cli claude \
  --optimization-mode leaderboard \
  --workspace-base /path/to/vector-add-campaign \
  --max-iters 5
```

Use the installed revision's `aka optimize --repo-root REPO --help` before launch.
The example uses the default exposed-Shape flow. A Production run with
`--no-expose-evaluator-shapes` may generate `agent_problem.json` before setup.

## Recorded reference baseline

[`metadata.json`](metadata.json) is an unchanged copy of the measured baseline
from job `ev_1861beac3248` on 2026-11-21 UTC: NVIDIA L20D, all shapes passed.
The candidate was this exact PyTorch `reference.py`, computing
`x + alpha * residual` over contiguous bf16 vectors. It was not an optimized
Triton or production kernel; `production_performance` is the existing metadata
field name, not a claim about the source's quality.

| Shape ID | numel | alpha | Measured latency (microseconds) |
| --- | ---: | ---: | ---: |
| 0 | 1024 | 1.0 | 9.456000290811062 |
| 1 | 65536 | 0.5 | 11.215999722480774 |
| 2 | 1048576 | 2.0 | 11.264000087976456 |

The worker reported torch `2.9.0+cu130`, Triton `3.7.1`, Python `3.12.3`, driver
`580.105.08`, and `clock_locked=false`. Its runtime was installed as
`atrex-bench==0.1.0.post20261111022558` (`source=job_install`), with evaluator
SHA-256 `183b6288b85f0cd7ee3828c1a2a9db922f8971774e0122f5fceccfb58e09456e`.
The local Task/runtime layout came from
[`alibaba/atrex-bench@1fa1f0d8a663093a22946270e7d6accb00c33698`](https://github.com/alibaba/atrex-bench/tree/1fa1f0d8a663093a22946270e7d6accb00c33698);
that local revision does not identify the remote wheel's Git revision.

| File measured or retained | SHA-256 |
| --- | --- |
| `reference.py` | `d55b1beb21b3e27bbe7a91c2b2558ea74187c5e3f798df48a0efb084be051c0c` |
| `input.py` | `f46bfc0b61306e2bb37653142ee7ef48c1af1870f7de6c55d81e1c0f52a90e76` |
| `shapes.json` | `314314866671d5b1ac7655178a23ec23a0d6d533afb0c8364b2f134529145d34` |
| `metadata.json` | `17d5b6e61678e635bb168f3afa7091237709f5fbf22f7d30c7c13f4eefa9f740` |

For another device, changed inputs or a fresh comparison, follow the ordinary
[Agate baseline commands](../../../../docs/schema-workflow.md#native-benchmark-inputs)
and keep the original job receipt. Record each actual Shape timing at
`shapes[ID].production_performance.performance_us`, `perf_gpu`, and the new
measurement provenance in `metadata.json`. Do not reuse these L20D timings as
measurements of a different workload or hardware.
