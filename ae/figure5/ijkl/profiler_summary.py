import json
from pathlib import Path


JSON_PATH = Path(
    "/home/zyzhao/LLMCompass/LLMCompass/ae/figure5/ijkl/transformer_CIM_opt_sim_profiler.json"
)


def bytes_to_gb(x):
    return x / 1e9


def cycles_to_billion(x):
    return x / 1e9


def safe_get(d, keys, default=0):
    cur = d
    for k in keys:
        if cur is None:
            return default
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur if cur is not None else default


def shape_to_str(shape):
    if shape is None:
        return "-"

    if not isinstance(shape, dict):
        return str(shape)

    if "K" in shape:
        return f'{shape.get("M", "-")} x {shape.get("N", "-")} x {shape.get("K", "-")}'
    elif "N" in shape:
        return f'{shape.get("M", "-")} x {shape.get("N", "-")}'
    elif "M" in shape:
        return f'{shape.get("M", "-")}'
    else:
        return str(shape)


def collect_valid_records(data):
    records = {}

    for name, record in data.items():
        if record is None:
            continue

        if isinstance(record, dict) and record.get("record", "not_null") is None:
            continue

        if not isinstance(record, dict):
            continue

        records[name] = record

    return records


def summarize_record(name, r):
    return {
        "name": name,
        "shape": shape_to_str(r.get("layer_shape")),
        "core_count": r.get("core_count", "-"),
        "tflop": float(r.get("tflop_count", 0) or 0),
        "total_latency": float(r.get("total_latency_cycles", 0) or 0),
        "dram_read": float(safe_get(r, ["dram_bytes", "read"], 0)),
        "dram_write": float(safe_get(r, ["dram_bytes", "write"], 0)),
        "l2_l1_read": float(safe_get(r, ["l2_to_l1_bytes", "read"], 0)),
        "l2_l1_write": float(safe_get(r, ["l2_to_l1_bytes", "write"], 0)),
        "dram_latency": float(r.get("dram_latency_cycles", 0) or 0),
        "l2_l1_latency": float(r.get("l2_to_l1_latency_cycles", 0) or 0),
        "compute_latency": float(r.get("compute_latency_cycles", 0) or 0),
        "profiler_strategy": r.get("profiler_strategy", "-"),
        "effective_scale": r.get("effective_scale", "-"),
    }


def print_line(char="-", n=120):
    print(char * n)


def print_table(headers, rows, aligns=None):
    if aligns is None:
        aligns = ["left"] * len(headers)

    str_rows = []
    for row in rows:
        str_rows.append([str(x) for x in row])

    widths = []
    for i, h in enumerate(headers):
        max_w = len(str(h))
        for row in str_rows:
            max_w = max(max_w, len(row[i]))
        widths.append(max_w)

    def fmt_cell(s, w, align):
        if align == "right":
            return s.rjust(w)
        return s.ljust(w)

    header_line = " | ".join(
        fmt_cell(str(headers[i]), widths[i], aligns[i]) for i in range(len(headers))
    )
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))

    print(header_line)
    print(sep_line)

    for row in str_rows:
        print(
            " | ".join(
                fmt_cell(row[i], widths[i], aligns[i]) for i in range(len(headers))
            )
        )


def main():
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Cannot find JSON file: {JSON_PATH}")

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    records = collect_valid_records(data)
    summaries = [summarize_record(name, r) for name, r in records.items()]

    total_tflop = sum(x["tflop"] for x in summaries)
    total_latency = sum(x["total_latency"] for x in summaries)
    total_dram_read = sum(x["dram_read"] for x in summaries)
    total_dram_write = sum(x["dram_write"] for x in summaries)
    total_l2_l1_read = sum(x["l2_l1_read"] for x in summaries)
    total_l2_l1_write = sum(x["l2_l1_write"] for x in summaries)
    total_dram_latency = sum(x["dram_latency"] for x in summaries)
    total_l2_l1_latency = sum(x["l2_l1_latency"] for x in summaries)
    total_compute_latency = sum(x["compute_latency"] for x in summaries)

    print_line("=")
    print("Overall Summary")
    print_line("=")

    overall_rows = [
        ["Total TFLOP", f"{total_tflop:.4f}"],
        ["Total latency", f"{cycles_to_billion(total_latency):.4f} B cycles"],
        ["Total DRAM read", f"{bytes_to_gb(total_dram_read):.4f} GB"],
        ["Total DRAM write", f"{bytes_to_gb(total_dram_write):.4f} GB"],
        ["Total L2 -> L1 read", f"{bytes_to_gb(total_l2_l1_read):.4f} GB"],
        ["Total L2 -> L1 write", f"{bytes_to_gb(total_l2_l1_write):.4f} GB"],
        ["Total DRAM latency", f"{cycles_to_billion(total_dram_latency):.4f} B cycles"],
        ["Total L2 -> L1 latency", f"{cycles_to_billion(total_l2_l1_latency):.4f} B cycles"],
        ["Total compute latency", f"{cycles_to_billion(total_compute_latency):.4f} B cycles"],
    ]

    if total_latency > 0:
        overall_rows += [
            ["DRAM latency ratio", f"{100 * total_dram_latency / total_latency:.2f}%"],
            ["L2 -> L1 latency ratio", f"{100 * total_l2_l1_latency / total_latency:.2f}%"],
            ["Compute latency ratio", f"{100 * total_compute_latency / total_latency:.2f}%"],
        ]

    print_table(["Metric", "Value"], overall_rows)

    print()
    print_line("=")
    print("Per-Operator Summary")
    print_line("=")

    summaries_sorted = sorted(summaries, key=lambda x: x["total_latency"], reverse=True)

    rows = []
    for x in summaries_sorted:
        latency_ratio = 100 * x["total_latency"] / total_latency if total_latency > 0 else 0
        compute_ratio = 100 * x["tflop"] / total_tflop if total_tflop > 0 else 0

        rows.append(
            [
                x["name"],
                x["shape"],
                x["core_count"],
                f"{x['tflop']:.4f}",
                f"{compute_ratio:.2f}%",
                f"{cycles_to_billion(x['total_latency']):.4f}",
                f"{latency_ratio:.2f}%",
                f"{bytes_to_gb(x['dram_read']):.4f}",
                f"{bytes_to_gb(x['dram_write']):.4f}",
                f"{bytes_to_gb(x['l2_l1_read']):.4f}",
                f"{bytes_to_gb(x['l2_l1_write']):.4f}",
            ]
        )

    print_table(
        [
            "Operator",
            "Shape",
            "Core",
            "TFLOP",
            "FLOP %",
            "Latency/Bcyc",
            "Lat %",
            "DRAM R/GB",
            "DRAM W/GB",
            "L2L1 R/GB",
            "L2L1 W/GB",
        ],
        rows,
        aligns=[
            "left",
            "left",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
        ],
    )

    print()
    print_line("=")
    print("Top Latency Operators")
    print_line("=")

    top_latency_rows = []
    for x in summaries_sorted[:5]:
        ratio = 100 * x["total_latency"] / total_latency if total_latency > 0 else 0
        top_latency_rows.append(
            [
                x["name"],
                f"{cycles_to_billion(x['total_latency']):.4f} B cycles",
                f"{ratio:.2f}%",
            ]
        )

    print_table(["Operator", "Latency", "Ratio"], top_latency_rows)

    print()
    print_line("=")
    print("Top DRAM Read Operators")
    print_line("=")

    top_dram = sorted(summaries, key=lambda x: x["dram_read"], reverse=True)

    rows = []
    for x in top_dram[:5]:
        ratio = 100 * x["dram_read"] / total_dram_read if total_dram_read > 0 else 0
        rows.append([x["name"], f"{bytes_to_gb(x['dram_read']):.4f} GB", f"{ratio:.2f}%"])

    print_table(["Operator", "DRAM Read", "Ratio"], rows)

    print()
    print_line("=")
    print("Top L2 -> L1 Read Operators")
    print_line("=")

    top_l2 = sorted(summaries, key=lambda x: x["l2_l1_read"], reverse=True)

    rows = []
    for x in top_l2[:5]:
        ratio = 100 * x["l2_l1_read"] / total_l2_l1_read if total_l2_l1_read > 0 else 0
        rows.append([x["name"], f"{bytes_to_gb(x['l2_l1_read']):.4f} GB", f"{ratio:.2f}%"])

    print_table(["Operator", "L2 -> L1 Read", "Ratio"], rows)

    print()
    print_line("=")
    print("Attention Summary")
    print_line("=")

    attention_names = [
        "Q_proj_x3_for_QKV",
        "Q_mul_K",
        "Softmax",
        "A_mul_V",
        "H_matmul0",
        "LayerNorm_MHA",
    ]

    attention = [x for x in summaries if x["name"] in attention_names]

    attn_tflop = sum(x["tflop"] for x in attention)
    attn_latency = sum(x["total_latency"] for x in attention)
    attn_dram_read = sum(x["dram_read"] for x in attention)
    attn_dram_write = sum(x["dram_write"] for x in attention)

    attn_rows = [
        ["Attention TFLOP", f"{attn_tflop:.4f}"],
        ["Attention latency", f"{cycles_to_billion(attn_latency):.4f} B cycles"],
        ["Attention latency ratio", f"{100 * attn_latency / total_latency:.2f}%" if total_latency > 0 else "-"],
        ["Attention DRAM read", f"{bytes_to_gb(attn_dram_read):.4f} GB"],
        ["Attention DRAM write", f"{bytes_to_gb(attn_dram_write):.4f} GB"],
    ]

    print_table(["Metric", "Value"], attn_rows)

    print()
    print("Attention operator breakdown:")

    rows = []
    for x in sorted(attention, key=lambda y: y["total_latency"], reverse=True):
        ratio = 100 * x["total_latency"] / attn_latency if attn_latency > 0 else 0
        rows.append(
            [
                x["name"],
                f"{x['tflop']:.4f}",
                f"{cycles_to_billion(x['total_latency']):.4f}",
                f"{ratio:.2f}%",
            ]
        )

    print_table(["Operator", "TFLOP", "Latency/Bcyc", "Attention Lat %"], rows)

    print()
    print_line("=")
    print("FFN Summary")
    print_line("=")

    ffn_names = [
        "LayerNorm_FFN",
        "H_matmul1",
        "Activation",
        "H_matmul2",
    ]

    ffn = [x for x in summaries if x["name"] in ffn_names]

    ffn_tflop = sum(x["tflop"] for x in ffn)
    ffn_latency = sum(x["total_latency"] for x in ffn)
    ffn_dram_read = sum(x["dram_read"] for x in ffn)
    ffn_dram_write = sum(x["dram_write"] for x in ffn)

    ffn_rows = [
        ["FFN TFLOP", f"{ffn_tflop:.4f}"],
        ["FFN latency", f"{cycles_to_billion(ffn_latency):.4f} B cycles"],
        ["FFN latency ratio", f"{100 * ffn_latency / total_latency:.2f}%" if total_latency > 0 else "-"],
        ["FFN DRAM read", f"{bytes_to_gb(ffn_dram_read):.4f} GB"],
        ["FFN DRAM write", f"{bytes_to_gb(ffn_dram_write):.4f} GB"],
    ]

    print_table(["Metric", "Value"], ffn_rows)

    print()
    print("FFN operator breakdown:")

    rows = []
    for x in sorted(ffn, key=lambda y: y["total_latency"], reverse=True):
        ratio = 100 * x["total_latency"] / ffn_latency if ffn_latency > 0 else 0
        rows.append(
            [
                x["name"],
                f"{x['tflop']:.4f}",
                f"{cycles_to_billion(x['total_latency']):.4f}",
                f"{ratio:.2f}%",
            ]
        )

    print_table(["Operator", "TFLOP", "Latency/Bcyc", "FFN Lat %"], rows)

    print()
    print_line("=")
    print("Bottleneck Conclusion")
    print_line("=")

    top4_latency = sum(x["total_latency"] for x in summaries_sorted[:4])
    top4_ratio = 100 * top4_latency / total_latency if total_latency > 0 else 0

    print(f"Top-4 latency operators occupy {top4_ratio:.2f}% of total latency.")
    print(
        "If this ratio is very high, the main optimization target should be these large matmul operators."
    )

    if total_latency > 0:
        dram_ratio = 100 * total_dram_latency / total_latency
        print(f"DRAM latency occupies {dram_ratio:.2f}% of total latency.")
        if dram_ratio > 50:
            print(
                "The profiling result is likely memory-bound. "
                "Reducing external memory access or improving data reuse should be prioritized."
            )
        else:
            print(
                "The profiling result is not strongly dominated by DRAM latency. "
                "Compute or on-chip transfer may also need optimization."
            )


if __name__ == "__main__":
    main()