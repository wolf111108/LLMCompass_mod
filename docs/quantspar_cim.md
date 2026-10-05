# Quantspar-compatible CIM GEMM backend

Default `CIMMacro.cim_backend = "quantspar"` applies to `heuristic-CIM`,
`heuristic-CIM-decode` and `heuristic-CIM-GQA-decode`. Set it to `"legacy"` to
restore prior behavior. Explicit weight-major/activation-major prefill modes
remain legacy alternatives. Vector operators are unchanged.

## Compute contract

Reference: quantspar main `ca07eb0fb543da70600cfbe60903462769d260f9`,
`quant/mapping.py`. Prefill layout selection is copied from that revision;
decode already used the same search. The hardware geometry is passed explicitly.
Published quantspar callers still omit geometry arguments: for non-64x48/16-bank
hardware, fix those callers before exporting matching statistics.

For a single operand, the imported speedup is `dense_steps / sparse_steps`:

```
dense_steps = baseline_height * dense_bits * K_rounds * M_rounds * N_rounds
compute_cycles = ceil(dense_steps * cycles_per_effective_bit / speedup)
```

`Mapping_stat_dynamic` first takes the busiest bank, sums all token/K rounds
for each macro lane, then takes the busiest macro. Apply the ratio ONCE over the
operator epoch, rather than per memory tile with extra compute barriers. The
ratio must already include actual bank/macro imbalance and any exponent costs
for its source path. No extra exponent or sign charge is added here.

Defaults: pure FP8 **mantissa-only 0MMM**, 3 dense bits, 1 cycle/effective bit,
8-bit activation storage. Mixed precision needs its actual average dense width:
`10 * high_ratio + 3 * mid_ratio + 1 * low_ratio`; BF16 mantissa-only is 7.
The old `2.4` throughput calibration and 4-bit SMMM decode baseline are not used
by this backend. Legacy throughput attributes remain for legacy simulation.
Do not use a mixed-precision width and a speedup normalized to a different width.

### Known upstream baseline defect

`source` (default) preserves quantspar's fixed-height denominator, including its
small-K error. Example: K=128, banks=16, height=64 -> effective height 8. A
no-bit-skipping input can report 8x. Importing that RAW 8x with `source` restores
the reported sparse steps; it does not endorse 8x as a physical sparse benefit.

`effective` uses `min(height, ceil(K/banks))`. Use it ONLY after recomputing the
upstream ratios with this same corrected denominator. Pairing `effective` with
old raw ratios would undercount compute by another factor of 8 in this example.
Quantspar is not modified by this change. Its ordinary prefill 7-vs-3-bit defect
and missing mixed-precision decode collection remain upstream blockers.

## Memory and GQA limits

Quantspar exports compute steps, not system memory timing. This backend adds a
**serialized bandwidth estimate**, not a trace-validated pipeline model:
activation HBM reads, weight service, compute, GB activation/partial-sum traffic,
and final output writes. No compute/IO overlap is assumed. Weight service takes
the maximum of external-read and aggregate local-write bandwidth costs; M-lane
weight replication is accounted for. A full activation operand is cached only
when it and the current partial strip fit GB, otherwise activation reads repeat
per N round. Service chunks must fit at least one row. K-lane/K-round partials
are materialized in GB. No per-event startup latency or exact FIFO scheduling
is modeled; these totals are estimates, not guaranteed physical upper bounds.

Independent attention operands remain serialized. Shared-KV GQA packs Q rows
into M and reuses K/V across those rows. Imported decode ratios must use this
same grouping; expanded-head statistics are incompatible. Current published
quantspar mixed-precision path returns before decode timing collection, so it
cannot yet supply a verified mixed-precision decode manifest.

## Figure 10 usage

Run from repository root in the existing dependency environment:

```bash
python -m ae.figure10_qwen.test_latency \
  --input-lengths 256 --output-lengths 32 --cores 16 \
  --prefill-dense-bits 3 --decode-dense-bits 3 \
  --cycles-per-effective-bit 1 --quantspar-baseline source \
  --output-dir outputs/qwen_quantspar
```

This uses existing hard-coded operator ratios, whose provenance is UNVERIFIED.
It is a smoke/sensitivity estimate until actual matching statistics are imported.
Use `--cores 32` if that is the geometry used for upstream statistics.

`--speedups-json /path/to/stats.json` overrides bit widths, cycle calibration,
baseline policy and all GEMM speedups. Geometry, model dimensions, batch, GQA
mode and sampled contexts must match. Missing operator ratios fail rather than
falling back to hard-coded values. The report records the manifest SHA256 and
its source/workload metadata. An illustrative schema (NOT measured statistics):

```json
{
  "source_commit": "ca07eb0fb543da70600cfbe60903462769d260f9",
  "geometry": {"height": 64, "width": 48, "banks": 16, "macros": 16},
  "baseline": "source",
  "dense_bits": {"prefill": 3, "decode": 3},
  "cycles_per_effective_bit": 1,
  "activation_storage_bits": 8,
  "workload": {
    "d_model": 5120, "ffn_dim": 13824, "q_heads": 40, "kv_heads": 8,
    "batch_size": 1, "shared_kv_gqa": true,
    "prefill_lengths": [256], "decode_cache_lengths": [256, 286]
  },
  "speedups": {
    "prefill": {"Q_proj": 1, "K_proj": 1, "V_proj": 1,
      "Q_mul_K": 1, "A_mul_V": 1, "H_matmul0": 1,
      "Gate_proj": 1, "Up_proj": 1, "Down_proj": 1},
    "decode": {"Q_proj": 1, "K_proj": 1, "V_proj": 1,
      "Q_mul_K": 1, "A_mul_V": 1, "H_matmul0": 1,
      "Gate_proj": 1, "Up_proj": 1, "Down_proj": 1}
  }
}
```

One ratio per phase/operator is still an approximation averaged over layers and
contexts. A per-layer/context trace interface is not implemented. The manifest
checks provenance/configuration, not whether its data were actually measured.

## Verification

```
python -m unittest discover -s tests -v
```

Golden checks cover quantspar prefill/decode layouts, the small-K raw baseline
and corrected contract, operator-wide speedup application without changing
traffic, GQA reuse, explicit bit/cycle calibration and invalid manifests. Legacy
regressions explicitly select the legacy backend; Figure-10 tests exercise the
new default. The editing environment lacked Torch/SCALEsim: import-only shims
were used to execute analytical code, and reject GPU/systolic calls. Full model
inference and hardware timing validation were not performed.
