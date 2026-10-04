# Figure 5 GEMM-only metrics

Run from the repository root, for example:

```bash
python -m ae.figure5.ijkl.test_transformer --simcim --qwen \
  --array_height 64 --array_width 48 --Nbank 16 --core_count 16
# Add --init for prefill.
python -m unittest discover -s tests -p test_cim_gemm.py
```

`*_sim_profiler.json` and the new `*_sim_gemm.csv` use the same GEMM profiler records.
For decode, `gemm_tpot_ms = sum(block GEMM cycles) * n_layers / clock_freq * 1000`.
This includes GEMM data movement and compute. It excludes softmax, normalization,
activation, fixed wrapper/launch overhead, LM head and sampling. It is not end-to-end TPOT.
A one-token run is explicitly labeled decode. Multi-token totals multiply the
single-step result at a fixed KV length; TPOT does not multiply by output length.
Prefill has no TPOT. Missing GEMM records are listed in metadata and suppress TPOT.

Existing breakdown CSVs and `model.latency` retain their legacy semantics for old
plotting callers. Use the new GEMM CSV or JSON `gemm_tpot_ms` for this comparison;
`model.gemm_latency` in the experiment driver is the one-block GEMM time in seconds.
Console GEMM summaries and exported JSON share a collector. Previously committed
JSON results are historical and must be regenerated to reflect these changes.

Qwen shared-KV decode and prefill attention GEMMs are enabled by default.
`--no-shared-kv` is the repeated-KV ablation; `--no-include-prefill-attention`
omits prefill attention and produces a partial-GEMM report. These flags apply to
the Qwen paths that accept them. CLI arguments, exact per-op matrix shapes,
hardware configuration, source commit and tracked dirty state accompany new JSON.
Existing workload lengths and model dimensions are unchanged.

Traffic fields distinguish:

- `hbm_weight_read_bytes`: off-chip weight operand reads (KV operands for decode attention).
- `cim_weight_write_bytes`: local macro writes, including prefill replicas.
- `traffic_bytes`: scaled original counters, including activation multicast and partial sums.
- `weight_write_bytes`: retained as a legacy alias of HBM operand reads, not local writes.
- Activation read trips use activation-only HBM reads and a denominator with the
  same serialized batch/group scale. Unknown counters are null.

Decode layout uses actual core count, rows, columns and banks. Every K/N tail is
covered; K-parallel macro occupancy and partitioned activation delivery are counted.
All M×N accumulators must fit the global buffer for the K-major schedule. Cross-K
reduction models accumulator reads and writes through the global buffer, with
intra-round reduction reads after compute; reduction arithmetic is assumed
bandwidth-bound. A dedicated reduction network/ALU is not yet calibrated.

Physical prefill compute and macro capacity use configured geometry. Throughput
already counts multiply and add separately, including in the roofline model.
Decode dense serial bits scale relative to the existing four-plane calibration;
`sparsity_ratio` remains the legacy argument name for a positive effective speedup,
not a percentage. The existing 2.4 calibration, model-specific speedups, storage
precisions and weight-update overlap assumptions are retained. These fixes do not
establish a fully calibrated reproduction of the paper's INT4/FP8 timing.
