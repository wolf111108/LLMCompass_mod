from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
    TransformerBlockOPTInitComputationTP,
    TransformerBlockOPTAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
from hardware_model.system import system_dict
import argparse
import json  # add

def get_op_by_candidates(model, names):
    for name in names:
        if hasattr(model, name):
            return getattr(model, name)
    return None

def build_profiler_ops(model):
    return {
        "Q_proj_x3_for_QKV": (model.Q_proj, 3),
        "Q_mul_K": (model.Q_mul_K, 1),
        "A_mul_V": (model.A_mul_V, 1),
        "H_matmul0": (model.H_matmul0, 1),
        "H_matmul1": (model.H_matmul1, 1),
        "H_matmul2": (model.H_matmul2, 1),
        "Softmax": (model.A_softmax, 1),

        # GPT-3: layer_norm0 / layer_norm1
        # OPT: layer_norm_attn / layer_norm_ffn
        "LayerNorm_MHA": (
            get_op_by_candidates(model, ["layer_norm_attn", "layer_norm0"]),
            1,
        ),
        "LayerNorm_FFN": (
            get_op_by_candidates(model, ["layer_norm_ffn", "layer_norm1", "layer_norm0"]),
            1,
        ),

        # GPT-3: H_gelu
        # OPT: H_act
        "Activation": (
            get_op_by_candidates(model, ["H_act", "H_gelu"]),
            1,
        ),
    }

def print_matmul_profiler_stats(model):  # add
    matmul_ops = build_profiler_ops(model)  # add
    for name, (op, output_scale) in matmul_ops.items():  # add
        print(f"\n===== {name} profiler =====")  # add
        if getattr(op, "profiler", None) is None:  # add
            print("No profiler object")  # add
            continue  # add
        record = op.profiler.get_best_record()  # add
        if record is None:  # add
            print("No profiler record")  # add
            continue  # add
        mapping = record["mapping"]  # add
        profiler_scale = getattr(op, "profiler_scale", 1)  # add
        effective_scale = output_scale * profiler_scale  # add
        extra_latency_cycles = getattr(op, "profiler_extra_latency_cycles", 0) * output_scale  # add
        extra_dram_write_bytes = getattr(op, "profiler_extra_dram_write_bytes", 0) * output_scale  # add
        effective_total_latency_cycles = record["total_latency"] * effective_scale + extra_latency_cycles  # add
        print("layer_shape:", op.profiler.layer_shape)  # add
        print("profiler_strategy:", getattr(op, "profiler_strategy", "direct_matmul"))  # add
        print("effective_scale:", effective_scale)  # add
        print("raw_total_latency_cycles:", record["total_latency"])  # add
        print("effective_total_latency_cycles:", effective_total_latency_cycles)  # add
        print("raw_dram_bytes:", record["dram_bytes"])  # add
        print("effective_dram_bytes:", {"read": int(record["dram_bytes"]["read"] * effective_scale), "write": int(record["dram_bytes"]["write"] * effective_scale + extra_dram_write_bytes)})  # add
        print("dram_latency_cycles:", record["dram_latency_cycles"] * effective_scale + extra_latency_cycles)  # add
        print("l2_to_l1_bytes:", {"read": int(record["l2_to_l1_bytes"]["read"] * effective_scale), "write": int(record["l2_to_l1_bytes"]["write"] * effective_scale)})  # add
        print("l2_to_l1_latency_cycles:", record["l2_to_l1_latency_cycles"] * effective_scale)  # add
        print("compute_latency_cycles:", record["compute_latency_cycles"] * effective_scale)  # add
        print("other_stats:", record["other_stats"])  # add
        print("mapping:", mapping.__dict__ if mapping is not None else None)  # add
  # add
def dump_matmul_profiler_stats(model, json_path):  # add
    matmul_ops = build_profiler_ops(model)  # add
    output_data = {}  # add
    for name, (op, output_scale) in matmul_ops.items():  # add
        if getattr(op, "profiler", None) is None:  # add
            output_data[name] = {"record": None}  # add
            continue  # add
        record = op.profiler.get_best_record()  # add
        if record is None:  # add
            output_data[name] = {"record": None}  # add
            continue  # add
        mapping = record["mapping"]  # add
        profiler_scale = getattr(op, "profiler_scale", 1)  # add
        effective_scale = output_scale * profiler_scale  # add
        extra_latency_cycles = getattr(op, "profiler_extra_latency_cycles", 0) * output_scale  # add
        extra_dram_write_bytes = getattr(op, "profiler_extra_dram_write_bytes", 0) * output_scale  # add
        flop_count = None  # add
        if hasattr(op, "flop_count"):  # add
            flop_count = op.flop_count * effective_scale  # add

        output_data[name] = {  # add
            "core_coun": A100_system.device.compute_module.core_count,  # add
            "flop_count": int(flop_count) if flop_count is not None else None,  # add
            "tflop_count": flop_count / 1e12 if flop_count is not None else None,  # add
            "layer_shape": op.profiler.layer_shape,  # add
            "profiler_strategy": getattr(op, "profiler_strategy", "direct_matmul"),  # add
            "output_scale": output_scale,  # add
            "profiler_scale": profiler_scale,  # add
            "effective_scale": effective_scale,  # add
            "raw_total_latency_cycles": record["total_latency"],  # add
            "extra_latency_cycles": extra_latency_cycles,  # add
            "total_latency_cycles": record["total_latency"] * effective_scale + extra_latency_cycles,  # add
            "raw_dram_bytes": record["dram_bytes"],  # add
            "extra_dram_write_bytes": extra_dram_write_bytes,  # add
            "dram_bytes": {"read": int(record["dram_bytes"]["read"] * effective_scale), "write": int(record["dram_bytes"]["write"] * effective_scale + extra_dram_write_bytes)},  # add
            "raw_dram_latency_cycles": record["dram_latency_cycles"],  # add
            "dram_latency_cycles": record["dram_latency_cycles"] * effective_scale + extra_latency_cycles,  # add
            "raw_l2_to_l1_bytes": record["l2_to_l1_bytes"],  # add
            "l2_to_l1_bytes": {"read": int(record["l2_to_l1_bytes"]["read"] * effective_scale), "write": int(record["l2_to_l1_bytes"]["write"] * effective_scale)},  # add
            "raw_l2_to_l1_latency_cycles": record["l2_to_l1_latency_cycles"],  # add
            "l2_to_l1_latency_cycles": record["l2_to_l1_latency_cycles"] * effective_scale,  # add
            "raw_compute_latency_cycles": record["compute_latency_cycles"],  # add
            "compute_latency_cycles": record["compute_latency_cycles"] * effective_scale,  # add
            "other_stats": record["other_stats"],  # add
            "mapping": mapping.__dict__ if mapping is not None else None,  # add
        }  # add
    with open(json_path, "w", encoding="utf-8") as f:  # add
        json.dump(output_data, f, indent=4, ensure_ascii=False, default=str)  # add
    print(f"Profiler stats written to {json_path}")  # add
  # add
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="initial computation")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU")
    parser.add_argument("--simgpu", action="store_true", help="Enable simulation")
    parser.add_argument("--simtpu", action="store_true", help="Enable simulation")
    parser.add_argument("--roofline", action="store_true", help="use roofline")
    parser.add_argument("--opt", action="store_true", help="use OPT1.3B model")
    args = parser.parse_args()

    bs = 8
    s = 2048
    if args.init:
        print("Initial computation")
        if args.simgpu:
            if args.opt:
                d_model= 2048
                n_heads= 32
                ffn_dim= 8192
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
            _ = model(Tensor([bs, s, 12288], data_type_dict["fp16"]))
            if args.roofline:
                model.roofline_model(TPU_system)
                file_name = "transformer_TPUv3_roofline.csv"
            else:
                model.compile_and_simulate(TPU_system, compile_mode="heuristic-TPU")
                file_name = "transformer_TPUv3_sim.csv"
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
        output_token_length = 1024
        if args.simgpu:
            if args.opt:
                d_model= 2048
                n_heads= 32
                ffn_dim= 8192
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
            _ = model(
                Tensor([bs, 1, 12288], data_type_dict["fp16"]), s + output_token_length
            )
            if args.roofline:
                model.roofline_model(TPU_system)
                file_name = "transformerAR_TPUv3_roofline.csv"
            else:
                model.compile_and_simulate(TPU_system, compile_mode="heuristic-TPU")
                file_name = "transformerAR_TPUv3_sim.csv"
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
    if not args.roofline and (args.simgpu or args.simtpu):  # add
        print_matmul_profiler_stats(model)  # add
        profiler_file_name = file_name.replace(".csv", "_profiler.json")  # add
        dump_matmul_profiler_stats(model, f"ae/figure5/ijkl/{profiler_file_name}")  # add
