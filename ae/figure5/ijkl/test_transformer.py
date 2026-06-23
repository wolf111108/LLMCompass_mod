from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
    TransformerBlockOPTInitComputationTP,
    TransformerBlockOPTAutoRegressionTP,
    TransformerBlockQwen25InitComputationTP,
    TransformerBlockQwen25AutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
from hardware_model.system import System, system_dict
from hardware_model.compute_module import compute_module_dict, build_cim_compute_module
from hardware_model.device import Device
from hardware_model.io_module import IO_module_dict
from hardware_model.memory_module import memory_module_dict
from hardware_model.interconnect import interconnect_module_dict
from math import ceil
import argparse
import json

# Only these are Matmul operations (have K dimension and flop_count > 0)
MATMUL_OP_NAMES = [
    "Q_proj_x3_for_QKV",
    "Q_proj",
    "K_proj",
    "V_proj",
    "Q_mul_K",
    "A_mul_V",
    "H_matmul0",
    "H_matmul1",
    "H_matmul2",
    "Gate_proj",
    "Up_proj",
    "Down_proj",
]

# Decode阶段这些操作的"weight"输入实际上是KV cache（Q_mul_K的K, A_mul_V的V）。
# 在CIM profiling里它们的weight_write_bytes其实是KV cache访问，需要单独统计。
KV_CACHE_OP_NAMES = ["Q_mul_K", "A_mul_V"]


def format_bytes(b):
    """Format byte count to human-readable string."""
    if b >= 1e9:
        return f"{b/1e9:.2f} GB"
    elif b >= 1e6:
        return f"{b/1e6:.2f} MB"
    elif b >= 1e3:
        return f"{b/1e3:.2f} KB"
    else:
        return f"{b} B"

def get_op_by_candidates(model, names):
    for name in names:
        if hasattr(model, name):
            return getattr(model, name)
    return None


def build_cim_system(array_height=128, array_width=128, Nbank=8, core_count=16, input_word_size=1):
    """Build a CIM system with configurable parameters."""
    cim_compute = build_cim_compute_module(
        array_height=array_height,
        array_width=array_width,
        Nbank=Nbank,
        core_count=core_count,
        input_word_size = input_word_size,
        l2_size=2048*512,
        l2_bandwidth_per_cycle=core_count*array_width*1
    )
    cim_device = Device(
        cim_compute,
        IO_module_dict["CIM_IO"],
        memory_module_dict["A100_80GB"],
        wu_io_module=IO_module_dict["CIM_WU_IO"],
    )
    return System(cim_device, interconnect_module_dict["NVLinkV3_FC_4"])


def build_profiler_ops(model):
    ops = {
        "Q_mul_K": (model.Q_mul_K, 1),
        "A_mul_V": (model.A_mul_V, 1),
        "H_matmul0": (model.H_matmul0, 1),
        "Softmax": (model.A_softmax, 1),

        "LayerNorm_MHA": (
            get_op_by_candidates(
                model,
                ["rms_norm_attn", "layer_norm_attn", "layer_norm0"],
            ),
            1,
        ),
        "LayerNorm_FFN": (
            get_op_by_candidates(
                model,
                ["rms_norm_ffn", "layer_norm_ffn", "layer_norm1", "layer_norm0"],
            ),
            1,
        ),

        "Activation": (
            get_op_by_candidates(model, ["H_act", "H_gelu"]),
            1,
        ),
    }

    # Qwen: GQA + SwiGLU
    # Q/K/V shape 不一样，不能用 Q_proj * 3
    if (
        hasattr(model, "Gate_proj")
        and hasattr(model, "Up_proj")
        and hasattr(model, "Down_proj")
    ):
        ops["Q_proj"] = (model.Q_proj, 1)
        ops["K_proj"] = (model.K_proj, 1)
        ops["V_proj"] = (model.V_proj, 1)

        ops["Gate_proj"] = (model.Gate_proj, 1)
        ops["Up_proj"] = (model.Up_proj, 1)
        ops["Down_proj"] = (model.Down_proj, 1)

    # GPT / OPT: 普通 MHA + 两层 FFN
    # Q/K/V shape 一样，可以用 Q_proj * 3
    else:
        ops["Q_proj_x3_for_QKV"] = (model.Q_proj, 3)
        ops["H_matmul1"] = (model.H_matmul1, 1)
        ops["H_matmul2"] = (model.H_matmul2, 1)

    return ops


def print_matmul_profiler_stats(model, system=None, n_layers=1, output_token_length=1):
    """只打印Matmul操作的关键CIM统计信息。"""
    matmul_ops = build_profiler_ops(model)

    # 获取CIM硬件参数（用于计算weight写入byte）
    array_width = None
    clock_freq = None
    if system is not None:
        try:
            array_width = system.device.compute_module.core.cim_macro.array_width
        except AttributeError:
            pass
        try:
            clock_freq = system.device.compute_module.clock_freq
        except AttributeError:
            pass

    print("\n" + "=" * 80)
    print("Matmul操作统计 (仅Matmul)")
    print("=" * 80)

    total_latency_all = 0
    total_compute_all = 0
    total_dram_all = 0
    total_flop_all = 0
    total_weight_write_bytes_all = 0
    total_kv_cache_access_bytes_all = 0

    for name, (op, output_scale) in matmul_ops.items():
        if name not in MATMUL_OP_NAMES:
            continue  # 跳过非Matmul操作

        if getattr(op, "profiler", None) is None:
            continue
        record = op.profiler.get_best_record()
        if record is None:
            continue

        is_kv_cache_op = name in KV_CACHE_OP_NAMES

        mapping = record["mapping"]
        profiler_scale = getattr(op, "profiler_scale", 1)
        effective_scale = output_scale * profiler_scale
        extra_latency_cycles = getattr(op, "profiler_extra_latency_cycles", 0) * output_scale
        extra_dram_write_bytes = getattr(op, "profiler_extra_dram_write_bytes", 0) * output_scale

        effective_total_latency = record["total_latency"] * effective_scale + extra_latency_cycles
        effective_compute_latency = record["compute_latency_cycles"] * effective_scale
        effective_dram_latency = record["dram_latency_cycles"] * effective_scale + extra_latency_cycles
        compute_ratio = effective_compute_latency / effective_total_latency * 100 if effective_total_latency > 0 else 0

        layer_shape = op.profiler.layer_shape
        M = layer_shape.get("M", 0)
        N = layer_shape.get("N", 0)
        K = layer_shape.get("K", 0)

        # CIM weight写入byte: directly from simulate_cim profiler
        weight_write_cycles_raw = record["other_stats"].get("weight_write_cycles", 0)
        weight_write_bytes_raw = record["other_stats"].get("weight_write_bytes", 0)
        weight_write_bytes = weight_write_bytes_raw * effective_scale

        # Weight重写次数: knm loop order下，weight在M维度复用，只在(k,n)变化时重写
        # 总次数 = num_K_tiles * num_N_tiles * effective_scale
        if mapping is not None:
            num_K_tiles = ceil(K / mapping.l2_tile_K) if mapping.l2_tile_K > 0 else 1
            num_M_tiles = ceil(M / mapping.l2_tile_M) if mapping.l2_tile_M > 0 else 1
            num_N_tiles = ceil(N / mapping.l2_tile_N) if mapping.l2_tile_N > 0 else 1
            weight_write_count_raw = num_K_tiles * num_N_tiles
        else:
            weight_write_count_raw = 0
        weight_write_count = weight_write_count_raw * effective_scale

        # Activation往返次数: activation_read的总byte / 单次完整activation的byte
        dram_read_raw = record["dram_bytes"]["read"]
        dram_write_raw = record["dram_bytes"]["write"]
        effective_dram_read = dram_read_raw * effective_scale
        effective_dram_write = dram_write_raw * effective_scale + extra_dram_write_bytes

        # 单次完整input activation: M*K*word_size, 单次完整output activation: M*N*word_size
        core = system.device.compute_module.core
        word_size = core.systolic_array.input_word_size if getattr(core, 'systolic_array', None) is not None else core.cim_macro.input_word_size
        single_input_act_bytes = M * K * word_size if M > 0 and K > 0 else 1
        single_output_act_bytes = M * N * word_size if M > 0 and N > 0 else 1
        act_read_trips = effective_dram_read / single_input_act_bytes
        act_write_trips = effective_dram_write / single_output_act_bytes

        # 并行化因子
        other_stats = record["other_stats"]
        K_factor = other_stats.get("K_factor", 1)
        M_factor = other_stats.get("M_factor", 1)
        N_factor = other_stats.get("N_factor", 1)

        print(f"\n----- {name} -----")
        print(f"  Shape: M={M}, N={N}, K={K}")
        print(f"  策略: {getattr(op, 'profiler_strategy', 'N/A')}, effective_scale={effective_scale}")
        if is_kv_cache_op:
            print(f"  KV cache访问: {format_bytes(weight_write_bytes)} (raw: {format_bytes(weight_write_bytes_raw)}, cycles={weight_write_cycles_raw})")
        else:
            print(f"  Weight写入: {format_bytes(weight_write_bytes)} (raw: {format_bytes(weight_write_bytes_raw)}, cycles={weight_write_cycles_raw})")
        print(f"  重写次数: {weight_write_count:.0f} (raw: {weight_write_count_raw}, per complete weight: {1.0 * weight_write_count})")
        print(f"  Activation读取次数: {act_read_trips:.2f}x (input act), 写出次数: {act_write_trips:.2f}x (output act)")
        print(f"  DRAM read: {format_bytes(effective_dram_read)}, write: {format_bytes(effective_dram_write)}")
        print(f"  Mapping: K_factor={K_factor}, M_factor={M_factor}, N_factor={N_factor}")
        if mapping is not None:
            print(f"  Tile分割: L2=({mapping.l2_tile_M}, {mapping.l2_tile_N}, {mapping.l2_tile_K}), "
                  f"L1=({mapping.l1_tile_M}, {mapping.l1_tile_N}, {mapping.l1_tile_K}), "
                  f"loop={mapping.l2_loop_order}, double_buf={mapping.is_l2_double_buffering}")
        # FLOP count and TOPS
        op_flop = getattr(op, "flop_count", 0) * output_scale
        tops = 0.0
        if clock_freq and effective_total_latency > 0 and op_flop > 0:
            time_seconds = effective_total_latency / clock_freq
            tops = op_flop / time_seconds / 1e12

        print(f"  Latency (pipeline后): total={effective_total_latency:.0f} cycles, "
              f"compute={effective_compute_latency:.0f} cycles, "
              f"dram={effective_dram_latency:.0f} cycles")
        print(f"    compute占比={compute_ratio:.1f}% (compute/raw sum, pipeline中compute与DRAM read重叠)")
        if clock_freq and op_flop > 0:
            print(f"  TOPS: {tops:.2f} TOPS (flop={op_flop/1e9:.2f} GFLOP, "
                  f"latency={effective_total_latency/clock_freq*1e6:.2f} us, "
                  f"freq={clock_freq/1e9:.2f} GHz)")

        total_latency_all += effective_total_latency
        total_compute_all += effective_compute_latency
        total_dram_all += effective_dram_latency
        total_flop_all += op_flop
        if is_kv_cache_op:
            total_kv_cache_access_bytes_all += weight_write_bytes
        else:
            total_weight_write_bytes_all += weight_write_bytes

    print(f"\n{'=' * 80}")
    overall_compute_ratio = total_compute_all / total_latency_all * 100 if total_latency_all > 0 else 0
    overall_tops = 0.0
    if clock_freq and total_latency_all > 0 and total_flop_all > 0:
        overall_time_seconds = total_latency_all / clock_freq
        overall_tops = total_flop_all / overall_time_seconds / 1e12
    print(f"所有Matmul总计: latency={total_latency_all:.0f} cycles, "
          f"compute={total_compute_all:.0f} cycles, "
          f"dram={total_dram_all:.0f} cycles, "
          f"compute占比={overall_compute_ratio:.1f}%")
    print(f"所有Matmul总计: TOPS={overall_tops:.2f} TOPS (总FLOP={total_flop_all/1e9:.2f} GFLOP, "
          f"总延迟={total_latency_all/clock_freq*1e6:.2f} us)" if clock_freq else "所有Matmul总计: TOPS=N/A (无clock_freq)")
    print(f"所有Matmul总计: weight_write_bytes={format_bytes(total_weight_write_bytes_all)}")
    print(f"所有Matmul总计: kv_cache_access_bytes={format_bytes(total_kv_cache_access_bytes_all)}")

    # ========== 完整模型 decode 汇总 ==========
    if n_layers > 1 or output_token_length > 1:
        full_model_total_tokens = n_layers * output_token_length
        full_latency = total_latency_all * full_model_total_tokens
        full_flop = total_flop_all * full_model_total_tokens
        full_weight_write = total_weight_write_bytes_all * full_model_total_tokens
        full_kv_cache_access = total_kv_cache_access_bytes_all * full_model_total_tokens
        full_dram_read = 0
        full_dram_write = 0
        # 重新遍历一次收集dram
        for _name2, (_op2, _os2) in matmul_ops.items():
            if _name2 not in MATMUL_OP_NAMES:
                continue
            if getattr(_op2, "profiler", None) is None:
                continue
            _rec2 = _op2.profiler.get_best_record()
            if _rec2 is None:
                continue
            _ps2 = getattr(_op2, "profiler_scale", 1)
            _es2 = _os2 * _ps2
            _extra_dw = getattr(_op2, "profiler_extra_dram_write_bytes", 0) * _os2
            full_dram_read += _rec2["dram_bytes"]["read"] * _es2
            full_dram_write += _rec2["dram_bytes"]["write"] * _es2 + _extra_dw
        full_dram_read *= full_model_total_tokens
        full_dram_write *= full_model_total_tokens

        full_tops = 0.0
        if clock_freq and full_latency > 0 and full_flop > 0:
            full_tops = full_flop / (full_latency / clock_freq) / 1e12

        mode_label = "Prefill" if output_token_length <= 1 else "Decode"
        print(f"\n{'=' * 80}")
        print(f"完整模型 {mode_label} 汇总 (n_layers={n_layers}, output_token_length={output_token_length}, "
              f"总layer步数={full_model_total_tokens})")
        print(f"  总延迟: {full_latency:.0f} cycles ({full_latency/clock_freq*1e6:.2f} us, "
              f"{full_latency/clock_freq*1e3:.4f} ms)" if clock_freq else
              f"  总延迟: {full_latency:.0f} cycles")
        print(f"  总FLOP: {full_flop/1e9:.2f} GFLOP ({full_flop/1e12:.4f} TFLOP)")
        print(f"  总TOPS: {full_tops:.2f} TOPS" if clock_freq else "  总TOPS: N/A")
        print(f"  总 Weight写入: {format_bytes(full_weight_write)}")
        print(f"  总 KV cache访问: {format_bytes(full_kv_cache_access)}")
        print(f"  总 DRAM read: {format_bytes(full_dram_read)}")
        print(f"  总 DRAM write: {format_bytes(full_dram_write)}")
        if output_token_length <= 1:
            print(f"  注意: Prefill阶段, 单层统计 × n_layers")
        else:
            print(f"  注意: 此汇总假设所有decode步骤的KV cache长度相同(近似)")
        print("=" * 80)


def dump_matmul_profiler_stats(model, json_path, system=None, n_layers=1, output_token_length=1):
    """只dump Matmul操作的关键统计信息到JSON。"""
    matmul_ops = build_profiler_ops(model)
    output_data = {}
    core_count = None
    array_width = None
    clock_freq = None
    if system is not None:
        core_count = system.device.compute_module.core_count
        try:
            array_width = system.device.compute_module.core.cim_macro.array_width
        except AttributeError:
            pass
        try:
            clock_freq = system.device.compute_module.clock_freq
        except AttributeError:
            pass

    for name, (op, output_scale) in matmul_ops.items():
        if name not in MATMUL_OP_NAMES:
            continue  # 跳过非Matmul操作

        if getattr(op, "profiler", None) is None:
            output_data[name] = {"record": None}
            continue
        record = op.profiler.get_best_record()
        if record is None:
            output_data[name] = {"record": None}
            continue

        mapping = record["mapping"]
        profiler_scale = getattr(op, "profiler_scale", 1)
        effective_scale = output_scale * profiler_scale
        extra_latency_cycles = getattr(op, "profiler_extra_latency_cycles", 0) * output_scale
        extra_dram_write_bytes = getattr(op, "profiler_extra_dram_write_bytes", 0) * output_scale

        layer_shape = op.profiler.layer_shape
        M = layer_shape.get("M", 0)
        N = layer_shape.get("N", 0)
        K = layer_shape.get("K", 0)

        effective_total_latency = record["total_latency"] * effective_scale + extra_latency_cycles
        effective_compute_latency = record["compute_latency_cycles"] * effective_scale
        effective_dram_latency = record["dram_latency_cycles"] * effective_scale + extra_latency_cycles
        compute_ratio = effective_compute_latency / effective_total_latency if effective_total_latency > 0 else 0

        is_kv_cache_op = name in KV_CACHE_OP_NAMES

        # CIM weight写入byte (KV cache op时实为KV cache访问)
        weight_write_cycles_raw = record["other_stats"].get("weight_write_cycles", 0)
        weight_write_bytes_raw = record["other_stats"].get("weight_write_bytes", 0)
        if weight_write_bytes_raw == 0 and array_width:
            # fallback for older records
            weight_write_bytes_raw = weight_write_cycles_raw * array_width
        weight_write_bytes = weight_write_bytes_raw * effective_scale

        # Weight重写次数: knm loop order下，weight在M维度复用
        if mapping is not None:
            num_K_tiles_d = ceil(K / mapping.l2_tile_K) if mapping.l2_tile_K > 0 else 1
            num_N_tiles_d = ceil(N / mapping.l2_tile_N) if mapping.l2_tile_N > 0 else 1
            weight_write_count_raw = num_K_tiles_d * num_N_tiles_d
        else:
            weight_write_count_raw = 0
        weight_write_count = weight_write_count_raw * effective_scale

        # Activation往返次数
        effective_dram_read = record["dram_bytes"]["read"] * effective_scale
        effective_dram_write = record["dram_bytes"]["write"] * effective_scale + extra_dram_write_bytes
        core = system.device.compute_module.core
        word_size = core.systolic_array.input_word_size if getattr(core, 'systolic_array', None) is not None else core.cim_macro.input_word_size
        single_input_act_bytes = M * K * word_size if M > 0 and K > 0 else 1
        single_output_act_bytes = M * N * word_size if M > 0 and N > 0 else 1

        flop_count = None
        if hasattr(op, "flop_count"):
            flop_count = op.flop_count * output_scale

        other_stats = record["other_stats"]

        # TOPS calculation
        tops = None
        if clock_freq and effective_total_latency > 0 and flop_count and flop_count > 0:
            time_seconds = effective_total_latency / clock_freq
            tops = round(flop_count / time_seconds / 1e12, 4)

        output_data[name] = {
            "core_count": core_count,
            "flop_count": int(flop_count) if flop_count is not None else None,
            "layer_shape": layer_shape,
            "profiler_strategy": getattr(op, "profiler_strategy", "N/A"),
            "effective_scale": effective_scale,
            "is_kv_cache_op": is_kv_cache_op,
            # Weight写入 / KV cache访问
            "weight_write_bytes": int(weight_write_bytes),
            "weight_write_bytes_human": format_bytes(weight_write_bytes),
            "weight_write_cycles_raw": weight_write_cycles_raw,
            "weight_write_count": int(weight_write_count),
            "weight_write_count_raw": weight_write_count_raw,
            # Activation往返
            "activation_read_trips": round(effective_dram_read / single_input_act_bytes, 4),
            "activation_write_trips": round(effective_dram_write / single_output_act_bytes, 4),
            "dram_read_bytes": int(effective_dram_read),
            "dram_write_bytes": int(effective_dram_write),
            # Mapping方式
            "K_factor": other_stats.get("K_factor", 1),
            "M_factor": other_stats.get("M_factor", 1),
            "N_factor": other_stats.get("N_factor", 1),
            # Tile分割
            "tile_L2": [mapping.l2_tile_M, mapping.l2_tile_N, mapping.l2_tile_K] if mapping else None,
            "tile_L1": [mapping.l1_tile_M, mapping.l1_tile_N, mapping.l1_tile_K] if mapping else None,
            "loop_order": mapping.l2_loop_order if mapping else None,
            "double_buffering": mapping.is_l2_double_buffering if mapping else None,
            # Latency (pipeline后的total; compute和dram为raw sum, pipeline中compute与DRAM read重叠)
            "total_latency_cycles": effective_total_latency,
            "compute_latency_cycles": effective_compute_latency,
            "dram_latency_cycles": effective_dram_latency,
            "compute_ratio": round(compute_ratio, 4),
            # TOPS
            "clock_freq_Hz": clock_freq,
            "tops": tops,
        }

    # Add overall TOPS summary
    total_flops = sum(
        d.get("flop_count", 0) or 0 for d in output_data.values()
        if isinstance(d, dict)
    )
    total_latency = sum(
        d.get("total_latency_cycles", 0) or 0 for d in output_data.values()
        if isinstance(d, dict)
    )
    total_weight_write_bytes = sum(
        d.get("weight_write_bytes", 0) or 0 for d in output_data.values()
        if isinstance(d, dict) and not d.get("is_kv_cache_op", False)
    )
    total_kv_cache_access_bytes = sum(
        d.get("weight_write_bytes", 0) or 0 for d in output_data.values()
        if isinstance(d, dict) and d.get("is_kv_cache_op", False)
    )
    overall_tops = None
    if clock_freq and total_latency > 0 and total_flops > 0:
        overall_tops = round(total_flops / (total_latency / clock_freq) / 1e12, 4)
    output_data["_overall"] = {
        "total_flop_count": total_flops,
        "total_flop_count_GFLOP": round(total_flops / 1e9, 4),
        "total_latency_cycles": total_latency,
        "total_latency_us": round(total_latency / clock_freq * 1e6, 4) if clock_freq else None,
        "clock_freq_Hz": clock_freq,
        "tops": overall_tops,
        "total_weight_write_bytes": int(total_weight_write_bytes),
        "total_weight_write_bytes_human": format_bytes(total_weight_write_bytes),
        "total_kv_cache_access_bytes": int(total_kv_cache_access_bytes),
        "total_kv_cache_access_bytes_human": format_bytes(total_kv_cache_access_bytes),
    }

    # ========== 完整模型 decode 汇总 ==========
    full_model_total_tokens = n_layers * output_token_length
    full_latency = total_latency * full_model_total_tokens
    full_flops = total_flops * full_model_total_tokens
    full_weight_write = total_weight_write_bytes * full_model_total_tokens
    full_kv_cache_access = total_kv_cache_access_bytes * full_model_total_tokens
    # 收集总dram
    full_dram_read = sum(
        (d.get("dram_read_bytes", 0) or 0) for d in output_data.values()
        if isinstance(d, dict)
    ) * full_model_total_tokens
    full_dram_write = sum(
        (d.get("dram_write_bytes", 0) or 0) for d in output_data.values()
        if isinstance(d, dict)
    ) * full_model_total_tokens

    full_tops = None
    if clock_freq and full_latency > 0 and full_flops > 0:
        full_tops = round(full_flops / (full_latency / clock_freq) / 1e12, 4)

    mode_label = "Prefill" if output_token_length <= 1 else "Decode"
    output_data["_full_model_summary"] = {
        "mode": mode_label,
        "n_layers": n_layers,
        "output_token_length": output_token_length,
        "total_tokens_times_layers": full_model_total_tokens,
        "note": ("单层统计 × n_layers" if output_token_length <= 1
                 else "假设所有decode步骤的KV cache长度相同(近似, 使用最后一步的seq_len)"),
        "total_latency_cycles": full_latency,
        "total_latency_us": round(full_latency / clock_freq * 1e6, 4) if clock_freq else None,
        "total_latency_ms": round(full_latency / clock_freq * 1e3, 4) if clock_freq else None,
        "total_flop_count": int(full_flops),
        "total_flop_count_GFLOP": round(full_flops / 1e9, 4),
        "total_flop_count_TFLOP": round(full_flops / 1e12, 6),
        "tops": full_tops,
        "total_weight_write_bytes": int(full_weight_write),
        "total_weight_write_bytes_human": format_bytes(full_weight_write),
        "total_kv_cache_access_bytes": int(full_kv_cache_access),
        "total_kv_cache_access_bytes_human": format_bytes(full_kv_cache_access),
        "total_dram_read_bytes": int(full_dram_read),
        "total_dram_read_bytes_human": format_bytes(full_dram_read),
        "total_dram_write_bytes": int(full_dram_write),
        "total_dram_write_bytes_human": format_bytes(full_dram_write),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False, default=str)
    print(f"Profiler stats written to {json_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="initial computation")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU")
    parser.add_argument("--simgpu", action="store_true", help="Enable simulation")
    parser.add_argument("--simcim", action="store_true", help="Enable CIM simulation")
    parser.add_argument("--simtpu", action="store_true", help="Enable simulation")
    parser.add_argument("--roofline", action="store_true", help="use roofline")
    parser.add_argument("--opt", action="store_true", help="use OPT1.3B model")
    parser.add_argument("--qwen", action="store_true", help="use Qwen model")
    parser.add_argument("--array_height", type=int, default=128, help="CIM macro array height")
    parser.add_argument("--array_width", type=int, default=128, help="CIM macro array width")
    parser.add_argument("--Nbank", type=int, default=8, help="CIM macro number of banks")
    parser.add_argument("--core_count", type=int, default=16, help="Number of CIM cores")
    args = parser.parse_args()

    bs = 1
    # s = 2048
    # output_token_length = 2048  # decode阶段生成token数
    #for qwen
    s = 8192
    output_token_length = 8192  # decode阶段生成token数
    #
    opt_dim = 2048
    opt_ffndim = 8192
    opt_head = 32
    opt_layer = 24
    qwen_dim = 1536
    qwen_ffndim = 8960
    qwen_head = 12
    qwen_kv_head = 2
    qwen_layer = 28
    qwen_kv_partition_mode = 'replicate'
    #opt_dim = 512
    #opt_ffndim = 2048
    #opt_head = 8
    #opt_layer = 4
    if args.init:
        print("Initial computation")
        if args.simgpu:
            if args.opt:
                d_model= opt_dim
                n_heads= opt_head
                ffn_dim= opt_ffndim
                model = TransformerBlockOPTInitComputationTP(
                    d_model= d_model,
                    n_heads= n_heads,
                    ffn_dim= ffn_dim,
                    device_count= 1,
                    data_type=data_type_dict["fp16"],
                    do_layer_norm_before=True,
                    activation_function="Gelu",
                )
                file_prefix = "transformer_A100_opt"
            else:
                d_model= 12288
                n_heads= 96
                model = TransformerBlockInitComputationTP(
                    d_model= d_model,
                    n_heads= n_heads,
                    device_count= 1,
                    data_type=data_type_dict["fp16"],
                )
                file_prefix = "transformer_A100"

            A100_system = system_dict["A100_4_fp16"]
            current_system = A100_system

            _ = model(
                Tensor([bs, s, d_model], data_type_dict["fp16"])
            )

            if args.roofline:
                model.roofline_model(A100_system)
                file_name = f"{file_prefix}_roofline.csv"
            else:
                model.compile_and_simulate(
                    A100_system,
                    compile_mode="heuristic-GPU",
                )
                file_name = f"{file_prefix}_sim.csv"
        if args.simtpu:
            model = TransformerBlockInitComputationTP(
                d_model=12288,
                n_heads=96,
                device_count=8,
                data_type=data_type_dict["fp16"],
            )
            TPU_system = system_dict["TPUv3_8"]
            current_system = TPU_system
            _ = model(Tensor([bs, s, 12288], data_type_dict["fp16"]))
            if args.roofline:
                model.roofline_model(TPU_system)
                file_name = "transformer_TPUv3_roofline.csv"
            else:
                model.compile_and_simulate(TPU_system, compile_mode="heuristic-TPU")
                file_name = "transformer_TPUv3_sim.csv"
        if args.simcim:
            print(f"CIM config: array_height={args.array_height}, array_width={args.array_width}, Nbank={args.Nbank}, core_count={args.core_count}")
            if args.qwen:
                d_model = qwen_dim
                n_heads = qwen_head
                n_kv_heads = qwen_kv_head
                ffn_dim = qwen_ffndim

                model = TransformerBlockQwen25InitComputationTP(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    ffn_dim=ffn_dim,
                    device_count=1,
                    data_type=data_type_dict["int8"],
                    kv_partition_mode=qwen_kv_partition_mode,
                )
                CIM_system = build_cim_system(
                array_height=args.array_height,
                array_width=args.array_width,
                Nbank=args.Nbank,
                core_count=args.core_count,
                )
                current_system = CIM_system

                _ = model(Tensor([bs, s, d_model], data_type_dict["int8"]))

                if args.roofline:
                    model.roofline_model(CIM_system)
                    file_name = "transformer_CIM_qwen_roofline.csv"
                else:
                    model.compile_and_simulate(CIM_system, compile_mode="heuristic-CIM")
                    file_name = "transformer_CIM_qwen_sim.csv"
            elif args.opt:
                model = TransformerBlockOPTInitComputationTP(
                    d_model=opt_dim,
                    n_heads=opt_head,
                    ffn_dim=opt_ffndim,
                    device_count=1,
                    data_type=data_type_dict["int8"],
                    do_layer_norm_before=True,
                    activation_function="Gelu",
                )
                CIM_system = build_cim_system(
                    array_height=args.array_height,
                    array_width=args.array_width,
                    Nbank=args.Nbank,
                    core_count=args.core_count,
                )
                current_system = CIM_system
                _ = model(Tensor([bs, s, opt_dim], data_type_dict["int8"]))
                if args.roofline:
                    model.roofline_model(CIM_system)
                    file_name = "transformer_CIM_opt_roofline.csv"
                else:
                    model.compile_and_simulate(CIM_system, compile_mode="heuristic-CIM")
                    file_name = "transformer_CIM_opt_sim.csv"
            else:
                model = TransformerBlockInitComputationTP(
                    d_model=12288,
                    n_heads=96,
                    device_count=1,
                    data_type=data_type_dict["fp16"],
                )
                CIM_system = build_cim_system(
                    array_height=args.array_height,
                    array_width=args.array_width,
                    Nbank=args.Nbank,
                    core_count=args.core_count,
                )
                current_system = CIM_system
                _ = model(Tensor([bs, s, 12288], data_type_dict["fp16"]))
                if args.roofline:
                    model.roofline_model(CIM_system)
                    file_name = "transformer_CIM_roofline.csv"
                else:
                    model.compile_and_simulate(CIM_system, compile_mode="heuristic-CIM")
                    file_name = "transformer_CIM_sim.csv"
        if args.gpu:
            model = TransformerBlockInitComputationTP(
                d_model=12288,
                n_heads=96,
                device_count=4,
                data_type=data_type_dict["fp16"],
            )
            _ = model(Tensor([bs, s, 12288], data_type_dict["fp16"]))
            model.run_on_gpu()
    else:
        print("Auto-regression / Decode")
        if args.simgpu:
            if args.opt:
                d_model= opt_dim
                n_heads= opt_head
                ffn_dim= opt_ffndim
                model = TransformerBlockOPTAutoRegressionTP(
                    d_model= d_model,
                    n_heads= n_heads,
                    ffn_dim= ffn_dim,
                    device_count=1,
                    data_type=data_type_dict["fp16"],
                    do_layer_norm_before=True,
                    activation_function= "Gelu",
                )
                file_prefix = "transformerAR_A100_opt"
            else:
                d_model= 12288
                n_heads= 96
                model = TransformerBlockAutoRegressionTP(
                    d_model=d_model,
                    n_heads=n_heads,
                    device_count=1,
                    data_type=data_type_dict["fp16"],
                )
                file_prefix = "transformerAR_A100"

            A100_system = system_dict["A100_4_fp16"]
            current_system = A100_system

            _ = model(
                Tensor([bs, 1, d_model], data_type_dict["fp16"]),
                s + output_token_length,
            )

            if args.roofline:
                model.roofline_model(A100_system)
                file_name = f"{file_prefix}_roofline.csv"
            else:
                model.compile_and_simulate(
                    A100_system,
                    compile_mode="heuristic-GPU",
                )
                file_name = f"{file_prefix}_sim.csv"
        if args.simtpu:
            model = TransformerBlockAutoRegressionTP(
                d_model=12288,
                n_heads=96,
                device_count=8,
                data_type=data_type_dict["fp16"],
            )
            TPU_system = system_dict["TPUv3_8"]
            current_system = TPU_system
            _ = model(
                Tensor([bs, 1, 12288], data_type_dict["fp16"]), s + output_token_length
            )
            if args.roofline:
                model.roofline_model(TPU_system)
                file_name = "transformerAR_TPUv3_roofline.csv"
            else:
                model.compile_and_simulate(TPU_system, compile_mode="heuristic-TPU")
                file_name = "transformerAR_TPUv3_sim.csv"
        if args.simcim:
            print(f"CIM config: array_height={args.array_height}, array_width={args.array_width}, Nbank={args.Nbank}, core_count={args.core_count}")
            if args.qwen:
                d_model = qwen_dim
                n_heads = qwen_head
                n_kv_heads = qwen_kv_head
                ffn_dim = qwen_ffndim

                model = TransformerBlockQwen25AutoRegressionTP(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    ffn_dim=ffn_dim,
                    device_count=1,
                    data_type=data_type_dict["int8"],
                    kv_partition_mode=qwen_kv_partition_mode,
                )

                CIM_system = build_cim_system(
                    array_height=args.array_height,
                    array_width=args.array_width,
                    Nbank=args.Nbank,
                    core_count=args.core_count,
                )
                current_system = CIM_system

                _ = model(
                    Tensor([bs, 1, d_model], data_type_dict["int8"]),
                    s + output_token_length,
                )

                if args.roofline:
                    model.roofline_model(CIM_system)
                    file_name = "transformerAR_CIM_qwen_roofline.csv"
                else:
                    model.compile_and_simulate(CIM_system, compile_mode="heuristic-CIM")
                    file_name = "transformerAR_CIM_qwen_sim.csv"
            elif args.opt:
                d_model= opt_dim
                n_heads= opt_head
                ffn_dim= opt_ffndim
                model = TransformerBlockOPTAutoRegressionTP(
                    d_model=opt_dim,
                    n_heads=opt_head,
                    ffn_dim=opt_ffndim,
                    device_count=1,
                    data_type=data_type_dict["int8"],
                    do_layer_norm_before=True,
                    activation_function="Gelu",
                )
                CIM_system = build_cim_system(
                    array_height=args.array_height,
                    array_width=args.array_width,
                    Nbank=args.Nbank,
                    core_count=args.core_count,
                )
                current_system = CIM_system
                _ = model(
                    Tensor([bs, 1, opt_dim], data_type_dict["int8"]), s + output_token_length
                )
                if args.roofline:
                    model.roofline_model(CIM_system)
                    file_name = "transformerAR_CIM_opt_roofline.csv"
                else:
                    model.compile_and_simulate(CIM_system, compile_mode="heuristic-CIM")
                    file_name = "transformerAR_CIM_opt_sim.csv"
            else:
                model = TransformerBlockAutoRegressionTP(
                    d_model=12288,
                    n_heads=96,
                    device_count=1,
                    data_type=data_type_dict["fp16"],
                )
                CIM_system = build_cim_system(
                    array_height=args.array_height,
                    array_width=args.array_width,
                    Nbank=args.Nbank,
                    core_count=args.core_count,
                )
                current_system = CIM_system
                _ = model(
                    Tensor([bs, 1, 12288], data_type_dict["fp16"]), s + output_token_length
                )
                if args.roofline:
                    model.roofline_model(CIM_system)
                    file_name = "transformerAR_CIM_roofline.csv"
                else:
                    model.compile_and_simulate(CIM_system, compile_mode="heuristic-CIM")
                    file_name = "transformerAR_CIM_sim.csv"
        if args.gpu:
            model = TransformerBlockAutoRegressionTP(
                d_model=12288,
                n_heads=96,
                device_count=4,
                data_type=data_type_dict["fp16"],
            )
            _ = model(
                Tensor([bs, 1, 12288], data_type_dict["fp16"]), s + output_token_length
            )
            model.run_on_gpu()
    with open(f"ae/figure5/ijkl/{file_name}", "w") as f:
        if args.roofline:
            f.write(model.roofline_log)
        else:
            f.write(model.simluate_log)
    if not args.roofline and (args.simgpu or args.simtpu or args.simcim):
        # 完整模型参数: OPT-1.3B n_layers=24
        if args.opt:
            n_layers = opt_layer  # OPT-1.3B
        elif args.qwen:
            n_layers = qwen_layer  # Qwen-25B
        else:
            n_layers = 24  # 以OPT-1.3B为例，实际可以根据模型调整
        # decode: n_layers × output_token_length; prefill: n_layers × 1
        print_matmul_profiler_stats(model, current_system,
                                    n_layers=n_layers,
                                    output_token_length=output_token_length if not args.init else 1)
        profiler_file_name = file_name.replace(".csv", "_profiler.json")
        dump_matmul_profiler_stats(model, f"ae/figure5/ijkl/{profiler_file_name}", current_system,
                                   n_layers=n_layers,
                                   output_token_length=output_token_length if not args.init else 1)
