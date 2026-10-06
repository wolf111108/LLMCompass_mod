"""Quantspar ca07eb0 layout and operator-wide compute compatibility.

No activation data are reconstructed here. Imported speedups must describe
max(sum(round bank workloads)) with this layout, bit width and baseline policy.
Memory uses a conservative serialized bandwidth model, not quantspar timing.
"""
import math
from math import ceil

QUANTSPAR_COMMIT = "ca07eb0fb543da70600cfbe60903462769d260f9"

def compute_optimal_macro_layout_prefill(
    Nmacro: int,
    in_features: int,
    out_features: int,
    seq_length: int,
    h: int = 64,
    w: int = 48,
    Nadder: int = 16,
):
    if Nmacro <= 0:
        raise ValueError(
            f"Nmacro must be positive, got {Nmacro}"
        )

    if in_features <= 0:
        raise ValueError(
            f"in_features must be positive, got {in_features}"
        )

    if out_features <= 0:
        raise ValueError(
            f"out_features must be positive, got {out_features}"
        )

    if seq_length <= 0:
        raise ValueError(
            f"seq_length must be positive, got {seq_length}"
        )

    if h <= 0 or w <= 0 or Nadder <= 0:
        raise ValueError(
            f"h, w and Nadder must be positive, "
            f"got h={h}, w={w}, Nadder={Nadder}"
        )

    # ------------------------------------------------------------
    # 一个 Macro 在一个 K round 中可以覆盖的输入维度
    #
    # 16 banks × 64 dimensions = 1024 dimensions
    # ------------------------------------------------------------
    k_capacity_per_macro_round = Nadder * h

    # 完整 GEMM 在三个方向上至少需要多少个基础 tile
    total_K_tiles = math.ceil(
        in_features / k_capacity_per_macro_round
    )

    total_M_tiles = seq_length

    total_N_tiles = math.ceil(
        out_features / w
    )

    max_K_factor = min(
        Nmacro,
        total_K_tiles,
    )

    max_M_factor = min(
        Nmacro,
        total_M_tiles,
    )

    # N_factor 固定为 1
    N_factor = 1

    best_layout = None
    best_score = None

    for K_factor in range(1, max_K_factor + 1):
        for M_factor in range(1, max_M_factor + 1):

                macros_used = (
                    K_factor
                    * M_factor
                    * N_factor
                )

                if macros_used > Nmacro:
                    continue

                # ====================================================
                # 三个方向仍需顺序执行的轮数
                # ====================================================

                # K_factor 个 Macro group 同时处理不同 K slice
                K_rounds = math.ceil(
                    in_features
                    / (
                        K_factor
                        * k_capacity_per_macro_round
                    )
                )

                # M_factor 个 Macro 处理不同 token
                M_rounds = math.ceil(
                    seq_length / M_factor
                )

                # N_factor=1 固定, 输出通道不并行
                N_rounds = math.ceil(
                    out_features / w
                )

                # 粗粒度总串行轮数
                total_serial_rounds = (
                    K_rounds
                    * M_rounds
                    * N_rounds
                )

                # ====================================================
                # 计算三个维度上的 padding/utilization
                # ====================================================

                K_capacity = (
                    K_factor
                    * K_rounds
                    * k_capacity_per_macro_round
                )

                M_capacity = (
                    M_factor
                    * M_rounds
                )

                N_capacity = (
                    N_factor
                    * N_rounds
                    * w
                )

                K_utilization = (
                    in_features / K_capacity
                )

                M_utilization = (
                    seq_length / M_capacity
                )

                N_utilization = (
                    out_features / N_capacity
                )

                macro_utilization = (
                    macros_used / Nmacro
                )

                overall_utilization = (
                    K_utilization
                    * M_utilization
                    * N_utilization
                    * macro_utilization
                )

                # K 并行时会产生 K_factor 份 partial sum。
                # 这里先用一个无量纲的简单 penalty 做次级比较。
                # 真正计算 latency 时仍应使用精确 reduction cycles。
                K_reduction_penalty = (
                    0
                    if K_factor == 1
                    else (
                        (K_factor - 1)
                        * M_rounds
                        * N_rounds
                    )
                )

                # ====================================================
                # 评分
                #
                # Python tuple 按顺序比较：
                #   1. 总串行轮数越小越好
                #   2. K reduction 越少越好
                #   3. 综合利用率越高越好
                #   4. 同条件优先 K > M > N
                # ====================================================
                score = (
                    total_serial_rounds,
                    # K_reduction_penalty,
                    # -overall_utilization,
                    -K_factor,
                    -M_factor,
                    -N_factor,
                )

                if best_score is None or score < best_score:
                    best_score = score

                    best_layout = {
                        "K_factor": K_factor,
                        "M_factor": M_factor,
                        "N_factor": N_factor,
                        "K_rounds": K_rounds,
                        "M_rounds": M_rounds,
                        "N_rounds": N_rounds,
                        "macros_used": macros_used,
                        "num_groups": (
                            K_factor * N_factor
                        ),
                        "macros_per_group": M_factor,
                        "K_utilization": K_utilization,
                        "M_utilization": M_utilization,
                        "N_utilization": N_utilization,
                        "macro_utilization": macro_utilization,
                        "overall_utilization": overall_utilization,
                        "total_serial_rounds": total_serial_rounds,
                        "K_reduction_penalty": K_reduction_penalty,
                    }

    if best_layout is None:
        raise RuntimeError(
            "Unable to find a valid macro layout"
        )

    return (
        best_layout["K_factor"],
        best_layout["M_factor"],
        best_layout["N_factor"],
        best_layout["K_rounds"],
        best_layout["M_rounds"],
        best_layout["N_rounds"],
    )

def compute_cycles(layout, *, height, banks, k, bits, cycles_per_bit=1.0,
                   speedup=1.0, baseline="source"):
    """Convert an operator-wide dense/sparse ratio to compute cycles.

    source reproduces Mapping_stat_dynamic's fixed-h denominator (including
    its known small-K error). effective requires speedups recomputed with
    h_eff in the denominator. Never mix ratios from these two policies.
    """
    for name, value in [("bits", bits), ("cycles_per_bit", cycles_per_bit),
                        ("speedup", speedup)]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if baseline not in ("source", "effective"):
        raise ValueError("baseline must be source or effective")
    _, _, _, kr, mr, nr = layout
    h_eff = max(1, min(height, ceil(k / banks)))
    dense = (height if baseline == "source" else h_eff) * bits * kr * mr * nr
    return dense, dense * cycles_per_bit / speedup


def load_speedup_manifest(macro, cores, path, expected_workload=None):
    """Fail closed on mismatched hardware and missing provenance fields."""
    import json
    import hashlib
    from pathlib import Path
    raw = Path(path).read_bytes()
    doc = json.loads(raw)
    geometry = dict(height=macro.array_height, width=macro.array_width,
                    banks=macro.Nbank, macros=cores)
    if doc.get("geometry") != geometry:
        raise ValueError("quantspar manifest geometry does not match CIM hardware")
    baseline = doc.get("baseline")
    bits = doc.get("dense_bits", {})
    cycles = doc.get("cycles_per_effective_bit")
    speedups = doc.get("speedups", {})
    if not doc.get("source_commit") or not doc.get("workload"):
        raise ValueError("quantspar manifest requires source_commit and workload provenance")
    if expected_workload is not None:
        workload = doc["workload"]
        if not isinstance(workload, dict) or any(
            workload.get(key) != value for key, value in expected_workload.items()
        ):
            raise ValueError("quantspar manifest workload does not match requested model/contexts")
    storage_bits = float(doc.get("activation_storage_bits", 8.0))
    if not math.isfinite(storage_bits) or storage_bits <= 0:
        raise ValueError("invalid quantspar activation storage bits")
    transport = doc.get("transport")
    if transport is not None:
        for name in ("linear_weight_storage_bits", "kv_storage_bits", "local_linear_weight_storage_bits"):
            value = transport.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid quantspar transport: {name}")
    for phase in ("prefill", "decode"):
        try:
            compute_cycles((1,1,1,1,1,1), height=macro.array_height,
                           banks=macro.Nbank, k=macro.Nbank*macro.array_height,
                           bits=float(bits[phase]), cycles_per_bit=float(cycles), baseline=baseline)
            if not isinstance(speedups[phase], dict) or not speedups[phase]:
                raise ValueError("empty phase speedups")
            for value in speedups[phase].values():
                if not math.isfinite(float(value)) or float(value) <= 0:
                    raise ValueError("invalid speedup")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid quantspar manifest phase: {phase}") from exc
    macro.quantspar_baseline = baseline
    macro.quantspar_prefill_dense_bits = float(bits["prefill"])
    macro.quantspar_decode_dense_bits = float(bits["decode"])
    macro.quantspar_cycles_per_effective_bit = float(cycles)
    macro.quantspar_activation_storage_bits = storage_bits
    if transport is not None:
        macro.quantspar_linear_weight_storage_bits = transport["linear_weight_storage_bits"]
        macro.quantspar_kv_storage_bits = transport["kv_storage_bits"]
        macro.quantspar_local_linear_weight_storage_bits = transport["local_linear_weight_storage_bits"]
    else:
        for name in ("quantspar_linear_weight_storage_bits", "quantspar_kv_storage_bits",
                     "quantspar_local_linear_weight_storage_bits"):
            if hasattr(macro, name):
                delattr(macro, name)
    macro.quantspar_speedups = speedups
    macro.quantspar_manifest_sha256 = hashlib.sha256(raw).hexdigest()
    macro.quantspar_manifest_source_commit = doc["source_commit"]
    macro.quantspar_manifest_workload = doc["workload"]
