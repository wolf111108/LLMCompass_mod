# New-code shared-KV decode sweep with 0.775-B activation

- Date: 2026-10-04.
- New-code commit: `61738e9`.
- Activation storage: `0.775 byte/element` for attention and non-attention decode GEMMs.
- Dense serial compute bits remain the calibrated default; only activation storage width is changed.
- KV cache / attention operand element size: `1 byte`.- Shared-KV GQA is enabled.- Off-chip bandwidth: `1 TB/s`.
- Global Buffer: `1 MiB`.- GEMM-only TPOT is reported; it excludes softmax/norm/activation/launch/LM head/sampling.
- Five points completed. `s=262143` is capacity-infeasible because shared-KV Q_mul_K must retain `5×262144×1 B` psums, exceeding the Global Buffer.

| s | GEMM-only TPOT | TOPS | DRAM read | KV access | Weight |
|---:|---:|---:|---:|---:|---:|
| 8191 | 14.892144 ms | 2.3151 | 14.031401 GB | 0.805306 GB | 13.212058 GB |
| 16383 | 15.940080 ms | 2.6681 | 14.848897 GB | 1.610613 GB | 13.212058 GB |
| 32767 | 18.171888 ms | 3.2268 | 16.483889 GB | 3.221225 GB | 13.212058 GB |
| 65535 | 22.639344 ms | 4.0129 | 19.753874 GB | 6.442451 GB | 13.212058 GB |
| 131071 | 31.570416 ms | 4.9183 | 26.293842 GB | 12.884902 GB | 13.212058 GB |
