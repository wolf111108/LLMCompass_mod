# Qwen Figure-10-style E2E latency

From the repository root, using the existing LLMCompass environment:

```bash
python -m ae.figure10_qwen.test_latency \
  --input-lengths 256 512 1024 --output-lengths 1 32 128 \
  --sample-stride 64 --output-dir outputs/qwen_fig10_run1
python -m ae.figure10_qwen.plot_latency outputs/qwen_fig10_run1/report.json
```

Defaults match the current figure5 Qwen dimensions: hidden 5120, FFN 13824,
40 Q heads, 8 KV heads, 48 layers, batch 1. Hardware defaults: 16 macros,
64 rows, 48 columns, 16 banks, existing 1 MiB GB / CIM IO configuration.
Decode shared KV is on by default; `--no-shared-kv` disables it.
`--prefill-mode` selects the existing CIM prefill mapper. Full attention is
always simulated. Existing per-operator effective speedups and storage precision
are inherited, not recalibrated; INT8 graph metadata is not a claim that every
physical unit operates in INT8. The JSON records the actual macro configuration.

## Scope and interpretation

This follows the **operator scope** of official `ae/figure10`, not a checkpoint
execution benchmark. All block GEMMs, Softmax, two Norms, activation, and configured
launch overheads are counted. Single-device execution has no all-reduce cost.
New wrapper classes restore decode vector costs without altering the old figure5
GEMM-only entry points. Each sampled phase constructs a fresh model.

**Approximations:** LayerNorm substitutes for RMSNorm; GeLU substitutes for SiLU;
prefill uses existing expanded-head BMM; operator/layer times are serially summed.
Excluded: embedding, final norm, LM head, sampling, RoPE, residual add, gate-times-up,
explicit scale/mask, separate KV append latency, host/network/queueing.
Existing operator data-movement assumptions are retained, including intermediate
spills; no cross-operator fusion/residency optimization is claimed.

`--control-us 0` is the default **idealized, uncalibrated** controller assumption.
A supplied value is charged to each of 9 GEMMs + 4 vector operations per block.
It replaces inherited GPU/TPU launch constants only in this new entry point.
Use measured/RTL-derived overheads before claiming absolute hardware performance.

Prefill produces the first output token; generating G tokens needs G-1 decode
calls with cache length BEFORE insertion S..S+G-2. The decode graph attends to
cache_length+1. This removes the one-step ambiguity in the original continuous
integration. We sum linearly interpolated **integer steps**, including both sample
endpoints. `--sample-stride 1` is exact per-token simulation; stride 64 is a speed
tradeoff and can smooth hardware tile discontinuities. Compare to stride 1 near
mapping boundaries. Full-stack times multiply block times by the configured layer
count, assuming identical layers/effective speedups.

## Outputs

- `report.json`: assumptions, commit/dirty status, CLI and hardware configuration,
  per-block samples and operator breakdown, per-request results.
- `requests.csv`: prefill, decode, E2E, GEMM-only E2E and mean TPOT in ms.
- `latency.png`: optional heatmap; no automatic comparison to an unrelated GPT baseline.

`prefill_ms` is a stack-level TTFT proxy within this scope, not full serving TTFT.
For G=1, decode is zero and mean TPOT is null. Existing report directories are
rejected to avoid silently overwriting experiments. Outputs are not auto-committed.

## Validation

```bash
python -m unittest discover -s tests -p test_qwen_fig10.py
python -m unittest discover -s tests -p test_cim_gemm.py
```

The new tests cover one-token/endpoint conventions, interpolation against an
analytical linear sum, incomplete coverage rejection, positive vector costs,
breakdown conservation, and unchanged GEMM cycles when launch overhead changes.

Implementation verification in the editing environment used import-only shims for
unavailable Torch/SCALEsim (GPU/systolic execution was not tested); actual CIM and
vector analytical functions ran unchanged. Existing 6 and new 2 tests passed.
`validation_requests.csv` records the current 48-layer model at input 256/512 and
output 1/32/128 with stride 64, zero control overhead, default shared KV and hardware.
These are model estimates under the assumptions above, not measured latencies.

At input 256 / output 32, stride-64 decode accumulation differed from stride-1
by -0.00172% in this validation. This is one checked workload, not a general
interpolation-error bound.
