# New-code shared-KV decode bandwidth sweep

- Date: 2026-10-04.
- Source commit: `61738e9`.
- Shared-KV GQA is enabled.- KV cache / attention operand element size: `1 byte`.- Decode activation storage: `0.775 byte`.
- Dense serial compute bits remain the calibrated default.- Global Buffer: `1 MiB`.- Off-chip CIM IO and weight-update IO bandwidth are varied together.
- Points: `s=8191,16383,32767,65535,131071` at bandwidths `46,368,568,768 GB/s` (20 total runs).- `total_latency_ms` means GEMM-only TPOT; softmax/norm/activation/launch/LM head/sampling are excluded.

| s | Bandwidth | GEMM-only TPOT | TOPS | Compute ratio | DRAM read | KV access | Weight |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8191 | 46 GB/s | 306.269856 ms | 0.1126 | 0.2544% | 14.031401 GB | 0.805306 GB | 13.212058 GB |
| 8191 | 368 GB/s | 39.027840 ms | 0.8834 | 1.9966% | 14.031401 GB | 0.805306 GB | 13.212058 GB |
| 8191 | 568 GB/s | 25.577616 ms | 1.3479 | 3.0465% | 14.031401 GB | 0.805306 GB | 13.212058 GB |
| 8191 | 768 GB/s | 19.124976 ms | 1.8027 | 4.0744% | 14.031401 GB | 0.805306 GB | 13.212058 GB |
| 16383 | 46 GB/s | 324.599328 ms | 0.1310 | 0.2973% | 14.848897 GB | 1.610613 GB | 13.212058 GB |
| 16383 | 368 GB/s | 41.507712 ms | 1.0246 | 2.3251% | 14.848897 GB | 1.610613 GB | 13.212058 GB |
| 16383 | 568 GB/s | 27.260304 ms | 1.5602 | 3.5403% | 14.848897 GB | 1.610613 GB | 13.212058 GB |
| 16383 | 768 GB/s | 20.420976 ms | 2.0827 | 4.7260% | 14.848897 GB | 1.610613 GB | 13.212058 GB |
| 32767 | 46 GB/s | 361.393824 ms | 0.1623 | 0.4073% | 16.483889 GB | 3.221225 GB | 13.212058 GB |
| 32767 | 368 GB/s | 46.603008 ms | 1.2582 | 3.1585% | 16.483889 GB | 3.221225 GB | 13.212058 GB |
| 32767 | 568 GB/s | 30.761232 ms | 1.9062 | 4.7851% | 16.483889 GB | 3.221225 GB | 13.212058 GB |
| 32767 | 768 GB/s | 23.150064 ms | 2.5329 | 6.3584% | 16.483889 GB | 3.221225 GB | 13.212058 GB |
| 65535 | 46 GB/s | 434.987424 ms | 0.2089 | 0.5724% | 19.753874 GB | 6.442451 GB | 13.212058 GB |
| 65535 | 368 GB/s | 56.797440 ms | 1.5995 | 4.3839% | 19.753874 GB | 6.442451 GB | 13.212058 GB |
| 65535 | 568 GB/s | 37.766928 ms | 2.4055 | 6.5929% | 19.753874 GB | 6.442451 GB | 13.212058 GB |
| 65535 | 768 GB/s | 28.611312 ms | 3.1753 | 8.7027% | 19.753874 GB | 6.442451 GB | 13.212058 GB |
| 131071 | 46 GB/s | 582.169632 ms | 0.2667 | 0.7767% | 26.293842 GB | 12.884902 GB | 13.212058 GB |
| 131071 | 368 GB/s | 77.182080 ms | 2.0118 | 5.8585% | 26.293842 GB | 12.884902 GB | 13.212058 GB |
| 131071 | 568 GB/s | 51.774864 ms | 2.9990 | 8.7334% | 26.293842 GB | 12.884902 GB | 13.212058 GB |
| 131071 | 768 GB/s | 39.530736 ms | 3.9279 | 11.4384% | 26.293842 GB | 12.884902 GB | 13.212058 GB |
