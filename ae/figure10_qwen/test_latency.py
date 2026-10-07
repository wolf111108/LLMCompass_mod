"""Qwen/CIM request latency using the official Figure 10 operator scope.

Run from repository root: python -m ae.figure10_qwen.test_latency --help
"""
import argparse
import contextlib
import csv
import json
import math
import os
from pathlib import Path
import subprocess


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def sample_lengths(start, steps, stride):
    """KV lengths BEFORE inserting the current token, inclusive endpoints."""
    if steps == 0:
        return []
    end = start + steps - 1
    return sorted(set(range(start, end + 1, stride)) | {end})


def sum_interpolated(samples, start, steps, key):
    """Sum integer decode steps using piecewise-linear interpolation.

    Unlike continuous integration, this is exact for stride=1 and one-step runs.
    """
    if not steps:
        return 0.0
    points = sorted(samples)
    end = start + steps - 1
    if not points or points[0] > start or points[-1] < end:
        raise ValueError("decode samples do not cover the request")
    if len(points) == 1:
        return steps * samples[points[0]][key]
    total = 0.0
    for a, b in zip(points, points[1:]):
        lo, hi = max(start, a), min(end, b - 1)
        if hi < lo:
            continue
        slope = (samples[b][key] - samples[a][key]) / (b - a)
        total += (hi-lo+1) * (samples[a][key] + slope * ((lo+hi)/2-a))
    if end == points[-1]:
        total += samples[end][key]
    return total


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-lengths", nargs="+", type=positive, default=[256])
    p.add_argument("--output-lengths", nargs="+", type=positive, default=[32])
    p.add_argument("--sample-stride", type=positive, default=64)
    p.add_argument("--batch-size", type=positive, default=1)
    for flag, default in [("d-model",5120),("ffn-dim",13824),("q-heads",40),
                          ("kv-heads",8),("layers",48),("array-height",64),
                          ("array-width",48),("banks",16),("cores",16)]:
        p.add_argument("--"+flag, type=positive, default=default)
    p.add_argument("--prefill-mode", choices=["heuristic-CIM", "heuristic-CIM-weight-major",
                   "heuristic-CIM-activation-major"], default="heuristic-CIM")
    p.add_argument("--cim-backend", choices=["quantspar", "legacy"], default="quantspar")
    p.add_argument("--prefill-dense-bits", type=nonnegative, default=3.0)
    p.add_argument("--decode-dense-bits", type=nonnegative, default=3.0)
    p.add_argument("--cycles-per-effective-bit", type=nonnegative, default=1.0)
    p.add_argument("--quantspar-baseline", choices=["source", "effective"], default="source")
    p.add_argument("--speedups-json", type=Path,
                   help="strict quantspar geometry/bit/baseline manifest; overrides dense bit options")
    p.add_argument("--allow-context-extrapolation", action="store_true",
                   help="reuse manifest phase/operator ratios at new lengths; model and geometry must still match")
    p.add_argument("--no-shared-kv", action="store_true")
    p.add_argument("--control-us", type=nonnegative, default=0,
                   help="per operator launch, default 0 = idealized controller (not calibrated)")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/figure10_qwen"))
    return p


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.allow_context_extrapolation and not args.speedups_json:
        cli.error("--allow-context-extrapolation requires --speedups-json")
    if args.d_model % args.q_heads or args.q_heads % args.kv_heads:
        raise ValueError("d-model must divide into Q heads; Q heads must divide into KV groups")
    from software_model.qwen_fig10 import QwenFigure10Prefill, QwenFigure10Decode
    from software_model.utils import Tensor, data_type_dict
    from hardware_model.compute_module import Overhead
    from ae.figure5.ijkl.test_transformer import build_cim_system, collect_matmul_profiler_stats

    system = build_cim_system(args.array_height,args.array_width,args.banks,args.cores)
    macro = system.device.compute_module.core.cim_macro
    macro.cim_backend = args.cim_backend
    macro.quantspar_prefill_dense_bits = args.prefill_dense_bits
    macro.quantspar_decode_dense_bits = args.decode_dense_bits
    macro.quantspar_cycles_per_effective_bit = args.cycles_per_effective_bit
    macro.quantspar_baseline = args.quantspar_baseline
    if args.speedups_json:
        if args.cim_backend != "quantspar" or args.prefill_mode != "heuristic-CIM":
            raise ValueError("speedups manifest requires default quantspar prefill/decode backend")
        from software_model.quantspar_cim import load_speedup_manifest
        load_speedup_manifest(macro, args.cores, args.speedups_json, expected_workload={
            "d_model": args.d_model, "ffn_dim": args.ffn_dim,
            "q_heads": args.q_heads, "kv_heads": args.kv_heads,
            "batch_size": args.batch_size, "shared_kv_gqa": not args.no_shared_kv,
            "prefill_lengths": sorted(set(args.input_lengths)),
            "decode_cache_lengths": sorted({c for s in args.input_lengths for g in args.output_lengths
                                            for c in sample_lengths(s, g-1, args.sample_stride)}),
        }, allow_context_extrapolation=args.allow_context_extrapolation)
    system.device.compute_module.overhead = Overhead(*([args.control_us * 1e-6]*4))
    dtype = data_type_dict["int8"]
    model_args = dict(d_model=args.d_model, n_heads=args.q_heads, n_kv_heads=args.kv_heads,
                      ffn_dim=args.ffn_dim, device_count=1, data_type=dtype,
                      kv_partition_mode="replicate")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "report.json").exists():
        raise FileExistsError("output report exists; choose a new --output-dir")
    names = ["qkv", "qk", "pv", "o", "gate_up", "down", "softmax",
             "norm_attn", "norm_ffn", "activation", "allreduce_attn", "allreduce_ffn"]
    hardware = None

    def simulate(phase, length):
        nonlocal hardware
        model = (QwenFigure10Prefill(**model_args) if phase == "prefill" else
                 QwenFigure10Decode(**model_args, shared_kv_gqa=not args.no_shared_kv))
        with open(os.devnull,"w") as quiet, contextlib.redirect_stdout(quiet):
            if phase == "prefill":
                model(Tensor([args.batch_size,length,args.d_model],dtype))
                seconds = model.compile_and_simulate(system,args.prefill_mode)
            else:
                model(Tensor([args.batch_size,1,args.d_model],dtype),length)
                seconds = model.compile_and_simulate(system)
            profile = collect_matmul_profiler_stats(model,system,n_layers=1,phase=phase)
        if not profile["_metadata"]["complete_block_gemm"]:
            raise RuntimeError("incomplete GEMM profile")
        hardware = dict(profile["_metadata"]["hardware"])
        vector = system.device.compute_module.core.vector_unit
        hardware["vector_unit"] = {k:v for k,v in vars(vector).items() if isinstance(v,(str,int,float,bool))}
        hardware["vector_data_type"] = str(vector.data_type.name) if hasattr(vector.data_type,"name") else str(vector.data_type.word_size)
        hardware["overhead_seconds"] = vars(system.device.compute_module.overhead)
        hardware["cim_compute_contract"] = {
            k:v for k,v in vars(macro).items() if k.startswith("quantspar_") or k == "cim_backend"
        }
        gemm = profile["_full_model_summary"]["total_latency_cycles"] / system.device.compute_module.clock_freq
        breakdown = dict(zip(names,[float(x)*1000 for x in model.simluate_log.split(",")]))
        if not math.isfinite(seconds) or seconds < 0 or not math.isclose(sum(breakdown.values()),seconds*1000,rel_tol=1e-8):
            raise RuntimeError("invalid block latency breakdown")
        row = dict(phase=phase,length=length,block_total_ms=seconds*1000,
                   block_gemm_ms=gemm*1000,block_breakdown_ms=breakdown,
                   block_control_ms=13*args.control_us/1000,
                   block_vector_ms=sum(list(breakdown.values())[6:10])-4*args.control_us/1000,
                   block_communication_ms=sum(list(breakdown.values())[10:12]))
        print(f"{phase} length={length}: block={seconds*1000:.6f} ms", flush=True)
        return row

    prefills = {s:simulate("prefill",s) for s in sorted(set(args.input_lengths))}
    lengths = sorted({c for s in args.input_lengths for g in args.output_lengths
                      for c in sample_lengths(s,g-1,args.sample_stride)})
    decodes = {s:simulate("decode",s) for s in lengths}
    requests = []
    for s in sorted(set(args.input_lengths)):
        for g in sorted(set(args.output_lengths)):
            pre = args.layers * prefills[s]["block_total_ms"]
            decode = args.layers * sum_interpolated(decodes,s,g-1,"block_total_ms")
            gemm = args.layers * (prefills[s]["block_gemm_ms"] + sum_interpolated(decodes,s,g-1,"block_gemm_ms"))
            requests.append(dict(input_length=s,output_length=g,prefill_ms=pre,
                decode_ms=decode,e2e_ms=pre+decode,gemm_e2e_ms=gemm,
                mean_tpot_ms=decode/(g-1) if g>1 else None))
    def git(*cmd):
        return subprocess.check_output(["git",*cmd],text=True).strip()
    metadata = dict(scope="Figure-10 Transformer-stack E2E estimate; not measured generation",
        approximations=["LayerNorm for RMSNorm", "GeLU for SiLU", "serial block/operator schedule",
                        "imported operator speedups" if args.speedups_json else "unverified existing hard-coded operator speedups", "prefill uses existing expanded-head BMM",
                        "linear interpolation between decode samples"],
        excludes=["embedding", "final norm", "LM head", "sampling", "RoPE", "residual add",
                  "gate-times-up", "explicit attention scale/mask", "separate KV append latency",
                  "host, network and queueing"],
        token_convention="prefill produces first token; G-1 decode calls; cache lengths S..S+G-2",
        control_assumption="idealized zero control" if args.control_us==0 else "user-specified per-operator control",
        commit=git("rev-parse","HEAD"), dirty=bool(git("status","--porcelain")),
        parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},hardware=hardware)
    if getattr(macro, "quantspar_manifest_context_extrapolated", False):
        metadata["approximations"].append(
            "context extrapolation: source per-phase/operator speedups held fixed; "
            "target mapping and IO recomputed; no measured long-context sparsity")
    report = dict(metadata=metadata,requests=requests,prefill_samples=list(prefills.values()),decode_samples=list(decodes.values()))
    (args.output_dir/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    with (args.output_dir/"requests.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(requests[0]),lineterminator="\n")
        writer.writeheader(); writer.writerows(requests)
    print(json.dumps(requests,indent=2))


if __name__ == "__main__":
    main()
