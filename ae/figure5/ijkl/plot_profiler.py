import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
CLOCK_HZ = 0.3e9
MATMUL_OVERHEAD_SEC = 2.1e-5
OVERHEAD_SEC_BY_CSV_KEY = {  #add
    "Q_K_V": MATMUL_OVERHEAD_SEC,  #add
    "Q_mul_K": MATMUL_OVERHEAD_SEC,  #add
    "A_mul_V": MATMUL_OVERHEAD_SEC,  #add
    "Wo_proj": MATMUL_OVERHEAD_SEC,  #add
    "W1_proj": MATMUL_OVERHEAD_SEC,  #add
    "W2_proj": MATMUL_OVERHEAD_SEC,  #add
    "Softmax": 1.2e-5,  #add
    "LayerNorm_MHA": 4.5e-5,  #add
    "LayerNorm_FFN": 4.5e-5,  #add
    "Activation": 4.5e-5,  # add: OPT / generic activation
    "GeLU": 4.5e-5,  # add: backward compatibility
    "ReLU": 4.5e-5,  # add: if ReLU profiler is exported
}  #add

CATEGORIES = [
    "Q_K_V",
    "Q_mul_K",
    "A_mul_V",
    "Wo_proj",
    "W1_proj",
    "W2_proj",
    "Softmax",
    "LayerNorm_MHA",
    "LayerNorm_FFN",
    #"GeLU",
    "Activation",  # add: 这一列表示 FFN activation，OPT 里通常是 ReLU
    "AllReduce_MHA",
    "AllReduce_FFN",
]

OPS = [
    ("QKV", "Q_proj_x3_for_QKV", 3, "Q_K_V"),
    ("QK", "Q_mul_K", 1, "Q_mul_K"),
    ("AV", "A_mul_V", 1, "A_mul_V"),
    ("Wo", "H_matmul0", 1, "Wo_proj"),
    ("W1", "H_matmul1", 1, "W1_proj"),
    ("W2", "H_matmul2", 1, "W2_proj"),
    ("Softmax", "Softmax", 1, "Softmax"),  #add
    ("LN-MHA", "LayerNorm_MHA", 1, "LayerNorm_MHA"),  #add
    ("LN-FFN", "LayerNorm_FFN", 1, "LayerNorm_FFN"),  #add
    #("GeLU", "GeLU", 1, "GeLU"),  #add
    ("Activation", ["Activation", "GeLU", "ReLU"], 1, "Activation"),  # add
]


def load_json(filename):
    with open(BASE_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_if_exists(filename):
    path = BASE_DIR / filename
    if not path.exists():
        return None
    return load_json(filename)


def load_csv(filename):
    path = BASE_DIR / filename
    if not path.exists():
        return {}
    text = path.read_text().strip()
    if not text:
        return {}
    values = [float(x.strip()) for x in text.split(",")]
    return dict(zip(CATEGORIES, values))


def gb(value):
    return value / 1e9


def seconds(cycles):
    return cycles / CLOCK_HZ


def gbps(byte_count, cycle_count):
    if cycle_count == 0:
        return 0
    return byte_count / seconds(cycle_count) / 1e9


def estimate_flops(record, json_key, flop_scale):  #add
    shape = record["layer_shape"]  #add
    if all(key in shape for key in ["M", "N", "K"]):  #add
        return 2 * shape["M"] * shape["N"] * shape["K"] * flop_scale  #add
    if json_key in ["GeLU", "Activation", "ReLU"]:  # add
        return record["other_stats"].get("total_flop_count", 0) * flop_scale  #add
    return None  #add


def op_metrics(data, label, json_key, scale, csv_key, csv_values):
    record = data[json_key]
    if "effective_scale" in record:
        scale = 1
        flop_scale = record["effective_scale"]
        overhead_scale = record.get("output_scale", 1)
        other_stats_scale = record["effective_scale"]
    else:
        flop_scale = scale
        overhead_scale = scale
        other_stats_scale = scale
    if "core_coun" in record:
        core_count = record["core_coun"]
    else:
        core_count = None
    dram_read = record["dram_bytes"]["read"] * scale
    dram_write = record["dram_bytes"]["write"] * scale
    l2_read = record["l2_to_l1_bytes"]["read"] * scale
    l2_write = record["l2_to_l1_bytes"]["write"] * scale
    total_cycles = record["total_latency_cycles"] * scale
    dram_cycles = record["dram_latency_cycles"] * scale
    l2_cycles = record["l2_to_l1_latency_cycles"] * scale
    compute_cycles = record["compute_latency_cycles"] * scale
    core_cycles = record["other_stats"].get("core_compute_cycles_from_l2_tiles", 0) * other_stats_scale
    flops = record.get("flop_count")  # add
    if flops is None:  # add
        flops = estimate_flops(record, json_key, flop_scale)  # add
    overhead_sec = OVERHEAD_SEC_BY_CSV_KEY.get(csv_key, MATMUL_OVERHEAD_SEC) * overhead_scale  # add
    effective_tflops = None if flops is None else flops / seconds(total_cycles) / 1e12  # add

    return {
        "core_count": core_count,
        "flop_count": flops,  # add
        "gflop_count": None if flops is None else flops / 1e9,  # add
        "tflop_count": None if flops is None else flops / 1e12,  # add
        "label": label,
        "json_key": json_key,
        "scale": scale,
        "csv_key": csv_key,
        "csv_latency_s": csv_values.get(csv_key),
        "json_latency_s": seconds(total_cycles),
        "json_latency_with_overhead_s": seconds(total_cycles) + overhead_sec, #add
        "total_cycles": total_cycles,
        "dram_read_gb": gb(dram_read),
        "dram_write_gb": gb(dram_write),
        "dram_total_gb": gb(dram_read + dram_write),
        "dram_latency_s": seconds(dram_cycles),
        "dram_effective_gbps": gbps(dram_read + dram_write, dram_cycles),
        "l2_read_gb": gb(l2_read),
        "l2_write_gb": gb(l2_write),
        "l2_total_gb": gb(l2_read + l2_write),
        "l2_latency_s": seconds(l2_cycles),
        "l2_effective_gbps": gbps(l2_read + l2_write, l2_cycles),
        "compute_latency_s_recorded": seconds(compute_cycles),
        "core_compute_latency_s_from_l2_tiles": seconds(core_cycles),
        "effective_tflops_from_total_latency": effective_tflops, #add
        "mapping": record["mapping"],
    }

def select_json_key(data, json_key_candidates):  # add
    if isinstance(json_key_candidates, str):  # add
        json_key_candidates = [json_key_candidates]  # add

    for key in json_key_candidates:  # add
        if key in data and data[key] is not None and "total_latency_cycles" in data[key]:  # add
            return key  # add

    return None  # add

def collect_metrics(json_filename, csv_filename):
    data = load_json_if_exists(json_filename)
    if data is None:
        return None
    csv_values = load_csv(csv_filename)
    metrics = []  # add
    for label, json_key_candidates, scale, csv_key in OPS:  # add
        json_key = select_json_key(data, json_key_candidates)  # add
        if json_key is None:  # add
            continue  # add
        metrics.append(op_metrics(data, label, json_key, scale, csv_key, csv_values))  # add
    return {
        "json_filename": json_filename,
        "csv_filename": csv_filename,
        "metrics": metrics,
        "csv_values": csv_values,
    }


def plot_latency(dataset, filename, ylabel, scale):
    metrics = dataset["metrics"]
    if not metrics:
        return
    colors = sns.color_palette("flare_r", len(metrics))
    plt.figure(figsize=(max(4.5, 0.55 * len(metrics)), 2.8))  #add
    x_positions = list(range(len(metrics)))  #add
    for x, color, item in zip(x_positions, colors, metrics):  #add
        value = item["json_latency_with_overhead_s"] * scale  #add
        plt.bar(x, value, color=color, width=0.55)  #add
    plt.ylabel(ylabel)
    plt.xticks(x_positions, [item["label"] for item in metrics], rotation=35, ha="right")  #add
    # bottom = 0 #delete
    # for color, item in zip(colors, metrics): #delete
    #     value = item["json_latency_with_overhead_s"] * scale #delete
    #     plt.bar(1, value, bottom=bottom, color=color, label=item["label"], width=0.5) #delete
    #     bottom += value #delete
    # plt.xticks([1], ["Profiler\nMatmul"]) #delete
    # handles, labels = plt.gca().get_legend_handles_labels() #delete
    # plt.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1, 1.05)) #delete
    plt.tight_layout()
    plt.savefig(BASE_DIR / filename, bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close()


def plot_traffic(datasets):
    available = [dataset for dataset in datasets if dataset and dataset["metrics"]]
    if not available:
        return
    op_labels = [item["label"] for item in available[0]["metrics"]]  #add
    x_positions = list(range(len(op_labels)))  #add
    bar_width = 0.7 / len(available)  #add
    colors = sns.color_palette("flare_r", len(available))  #add
    plt.figure(figsize=(max(4.5, 0.55 * len(op_labels)), 2.8))  #add
    for dataset_index, dataset in enumerate(available):  #add
        metric_by_label = {item["label"]: item for item in dataset["metrics"]}  #add
        offset = (dataset_index - (len(available) - 1) / 2) * bar_width  #add
        values = [metric_by_label[label]["dram_total_gb"] if label in metric_by_label else 0 for label in op_labels]  #add
        plt.bar([x + offset for x in x_positions], values, width=bar_width, color=colors[dataset_index], label=dataset["title"])  #add
    plt.ylabel("DRAM Traffic (GB)")
    plt.xticks(x_positions, op_labels, rotation=35, ha="right")  #add
    if len(available) > 1:  #add
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1.05))  #add
    # colors = sns.color_palette("flare_r", len(OPS)) #delete
    # plt.figure(figsize=(max(3, 2.2 * len(available)), 2.8)) #delete
    # for x, dataset in enumerate(available, start=1): #delete
    #     metrics = dataset["metrics"] #delete
    #     bottom = 0 #delete
    #     for color, item in zip(colors, metrics): #delete
    #         value = item["dram_total_gb"] #delete
    #         plt.bar(x, value, bottom=bottom, color=color, label=item["label"] if x == 1 else None, width=0.5) #delete
    #         bottom += value #delete
    # plt.xticks(range(1, len(available) + 1), [dataset["title"].replace(" ", "\n") for dataset in available]) #delete
    # handles, labels = plt.gca().get_legend_handles_labels() #delete
    # plt.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1, 1.05)) #delete
    plt.tight_layout()
    plt.savefig(BASE_DIR / "profiler_dram_traffic.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close()


def write_summary(datasets):
    lines = []
    lines.append("A100 profiler summary")
    lines.append(f"Clock frequency: {CLOCK_HZ / 1e9:.3f} GHz")
    lines.append(f"Matmul overhead used by CSV: {MATMUL_OVERHEAD_SEC * 1e6:.1f} us per matmul call")
    lines.append("")
    available = [dataset for dataset in datasets if dataset and dataset["metrics"]]
    if not available:
        lines.append("No profiler JSON data found.")
        (BASE_DIR / "profiler_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines.append("CSV consistency check")
    for dataset in available:
        title = dataset["title"]
        metrics = dataset["metrics"]
        lines.append(f"[{title}]")
        for item in metrics:
            if item["csv_latency_s"] is None:
                lines.append(f"{item['label']}: JSON+overhead={item['json_latency_with_overhead_s']:.9f}s, CSV=unavailable")
            else:
                diff = item["json_latency_with_overhead_s"] - item["csv_latency_s"]
                lines.append(
                    f"{item['label']}: JSON+overhead={item['json_latency_with_overhead_s']:.9f}s, "
                    f"CSV={item['csv_latency_s']:.9f}s, diff={diff:.9e}s"
                )
        lines.append("")
    lines.append("Readable profiler data")
    for dataset in available:
        title = dataset["title"]
        metrics = dataset["metrics"]
        lines.append(f"[{title}]")
        for item in metrics:
            effective_tflops_text = "N/A" if item["effective_tflops_from_total_latency"] is None else f"{item['effective_tflops_from_total_latency']:.6f}"  #add
            flop_count_text = "N/A" if item["flop_count"] is None else str(int(item["flop_count"]))  # add
            gflop_count_text = "N/A" if item["gflop_count"] is None else f"{item['gflop_count']:.6f}"  # add
            tflop_count_text = "N/A" if item["tflop_count"] is None else f"{item['tflop_count']:.6f}"  # add

            lines.append(f"{item['label']}")
            lines.append(f"  source: {item['json_key']}")
            lines.append(f"  scale: {item['scale']}")
            lines.append(f"  core_count: {item['core_count']}")
            lines.append(f"  latency_without_overhead_s: {item['json_latency_s']:.9f}")
            lines.append(f"  latency_with_overhead_s: {item['json_latency_with_overhead_s']:.9f}")

            lines.append(f"  total_cycles: {item['total_cycles']:.0f}")
            lines.append(f"  flop_count: {flop_count_text}")  # add
            lines.append(f"  gflop_count: {gflop_count_text}")  # add
            lines.append(f"  tflop_count: {tflop_count_text}")  # add

            lines.append(f"  dram_total_GB: {item['dram_total_gb']:.6f}")
            lines.append(f"  dram_read_GB: {item['dram_read_gb']:.6f}")
            lines.append(f"  dram_write_GB: {item['dram_write_gb']:.6f}")
            lines.append(f"  dram_latency_s: {item['dram_latency_s']:.9f}")
            lines.append(f"  dram_effective_BW_GB_per_s: {item['dram_effective_gbps']:.6f}")
            lines.append(f"  l2_l1_total_GB: {item['l2_total_gb']:.6f}")
            lines.append(f"  l2_l1_read_GB: {item['l2_read_gb']:.6f}")
            lines.append(f"  l2_l1_write_GB: {item['l2_write_gb']:.6f}")
            lines.append(f"  l2_l1_latency_s: {item['l2_latency_s']:.9f}")
            lines.append(f"  l2_l1_effective_BW_GB_per_s: {item['l2_effective_gbps']:.6f}")
            lines.append(f"  recorded_compute_latency_s: {item['compute_latency_s_recorded']:.9f}")
            lines.append(f"  core_compute_from_l2_tiles_s: {item['core_compute_latency_s_from_l2_tiles']:.9f}")
            lines.append(f"  effective_total_TFLOP_per_s: {effective_tflops_text}")  #add
            lines.append("")
        lines.append("")
    lines.append("Notes")
    lines.append("1. JSON contains profiler records for matmul-family ops plus Softmax, LayerNorm, and GeLU when available. CSV also contains AllReduce.") #add
    lines.append("2. The JSON key Q_proj_x3_for_QKV is exported as effective QKV data by test_transformer.py.")
    lines.append("3. The CSV matmul entries include 21us kernel overhead per matmul call. JSON total_latency_cycles does not include that overhead; after adding it, the matching CSV entries agree.")
    lines.append("4. compute_latency_cycles is larger than total_latency_cycles for these mappings, and is roughly larger than core_compute_cycles_from_l2_tiles. Treat it as a diagnostic counter, not an additive latency component.")
    lines.append("5. DRAM/L2/core component counters are not expected to sum to total_latency because the simulator overlaps read/compute/write under double buffering.")
    (BASE_DIR / "profiler_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["auto", "prefill", "decode", "both"],
        default="auto",
        help="Choose which profiler datasets to plot. auto uses whichever JSON files exist.",
    )
    args = parser.parse_args()
    dataset_specs = [
        {
            "mode": "prefill",
            "title": "Prefill A100",
            "json_filename": "transformer_A100_sim_profiler.json",
            "csv_filename": "transformer_A100_sim.csv",
            "latency_filename": "profiler_latency_prefill.pdf",
            "latency_ylabel": "Latency (s)",
            "latency_scale": 1,
        },
        {
            "mode": "decode",
            "title": "Decode A100",
            "json_filename": "transformerAR_A100_sim_profiler.json",
            "csv_filename": "transformerAR_A100_sim.csv",
            "latency_filename": "profiler_latency_decode.pdf",
            "latency_ylabel": "Latency (ms)",
            "latency_scale": 1e3,
        },
                {
            "mode": "decode_opt",
            "title": "Decode A100 OPT",
            "json_filename": "transformerAR_A100_opt_sim_profiler.json",
            "csv_filename": "transformerAR_A100_opt_sim.csv",
            "latency_filename": "profiler_latency_decode_opt.pdf",
            "latency_ylabel": "Latency (ms)",
            "latency_scale": 1e3,
        },
                {
            "mode": "prefill_opt",
            "title": "Prefill A100 OPT",
            "json_filename": "transformer_A100_opt_sim_profiler.json",
            "csv_filename": "transformer_A100_opt_sim.csv",
            "latency_filename": "profiler_latency_prefill_opt.pdf",
            "latency_ylabel": "Latency (ms)",
            "latency_scale": 1e3,
        },
    ]
    if args.mode == "prefill":
        dataset_specs = [spec for spec in dataset_specs if spec["mode"] == "prefill"]
    elif args.mode == "decode":
        dataset_specs = [spec for spec in dataset_specs if spec["mode"] == "decode"]
    datasets = []
    for spec in dataset_specs:
        dataset = collect_metrics(spec["json_filename"], spec["csv_filename"])
        if dataset is None:
            datasets.append(None)
            print(f"Skipping missing profiler JSON: {spec['json_filename']}")
            continue
        dataset.update(spec)
        datasets.append(dataset)
        plot_latency(dataset, spec["latency_filename"], spec["latency_ylabel"], spec["latency_scale"])
    plot_traffic(datasets)
    write_summary(datasets)


if __name__ == "__main__":
    main()
