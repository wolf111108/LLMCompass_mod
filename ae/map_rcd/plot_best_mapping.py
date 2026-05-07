import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys


def load_data(filepath: str) -> pd.DataFrame:
    """Load JSONL file and return DataFrame."""
    records = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def plot_latency_bar(df: pd.DataFrame, output_path: str):
    """Horizontal bar chart: latency breakdown by layer."""
    # Aggregate latency by layer (sum, in case same layer appears multiple times)
    df_agg = df.groupby("layer_name")["latency_ms"].sum().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(6, 0.4 * len(df_agg) + 1.2))

    colors = sns.color_palette("flare", len(df_agg))
    bars = ax.barh(df_agg.index, df_agg.values, color=colors, height=0.6)

    # Annotate each bar with the latency value
    for bar, val in zip(bars, df_agg.values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    ax.set_xlabel("Latency (ms)", fontsize=10)
    ax.set_title("Matmul Latency Breakdown", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlim(0, df_agg.max() * 1.25)

    # Add total latency annotation
    total = df_agg.sum()
    ax.text(0.98, 0.02, f"Total: {total:.4f} ms", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close()
    print(f"[Saved] Latency bar chart -> {output_path}")


def plot_tiling_heatmap(df: pd.DataFrame, output_path: str):
    """Heatmap: tiling parameters (l2 & l1 tile sizes) by layer."""
    tile_cols = ["l2_tile_M", "l2_tile_N", "l2_tile_K",
                 "l1_tile_M", "l1_tile_N", "l1_tile_K"]

    # Use the first record per layer for tiling params (tiling is deterministic per layer)
    df_tile = df.groupby("layer_name")[tile_cols].first().reset_index()
    df_tile = df_tile.set_index("layer_name")

    # Rename columns for readability in heatmap
    rename_map = {
        "l2_tile_M": "L2 Tile M",
        "l2_tile_N": "L2 Tile N",
        "l2_tile_K": "L2 Tile K",
        "l1_tile_M": "L1 Tile M",
        "l1_tile_N": "L1 Tile N",
        "l1_tile_K": "L1 Tile K",
    }
    df_tile = df_tile.rename(columns=rename_map)

    fig, ax = plt.subplots(figsize=(0.5 * len(df_tile.columns) + 2,
                                     0.4 * len(df_tile) + 1.5))

    sns.heatmap(df_tile, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Tile Size", "shrink": 0.8},
                ax=ax)

    ax.set_title("Best Tiling Strategy by Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_xlabel("Tiling Parameter", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close()
    print(f"[Saved] Tiling heatmap -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot best mapping results from JSONL file."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the JSONL file containing best mapping records.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same directory as input file).",
    )
    args = parser.parse_args()

    # Validate input file
    if not os.path.isfile(args.input_file):
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory and base name
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_dir = args.output_dir
    else:
        out_dir = os.path.dirname(args.input_file) or "."

    base_name = os.path.splitext(os.path.basename(args.input_file))[0]

    latency_output = os.path.join(out_dir, f"{base_name}_latency.pdf")
    tiling_output = os.path.join(out_dir, f"{base_name}_tiling.pdf")

    # Load data
    print(f"Loading data from: {args.input_file}")
    df = load_data(args.input_file)
    print(f"Loaded {len(df)} records, {df['layer_name'].nunique()} unique layers")
    print(f"Columns: {list(df.columns)}")

    # Generate plots
    plot_latency_bar(df, latency_output)
    plot_tiling_heatmap(df, tiling_output)


if __name__ == "__main__":
    main()