#!/usr/bin/env python3
"""
Plot bandwidth results from vLLM KV Cache Offload Benchmark output.

This script reads the Markdown table printed by the benchmark script
and generates a bandwidth plot (GiB/s vs block size).

Usage:
    # Option 1: Pipe benchmark output directly
    python benchmark_swap_blocks.py [args] | tee results.txt
    python plot_from_table.py < results.txt

    # Option 2: Save output to file first, then plot
    python benchmark_swap_blocks.py [args] > results.txt
    python plot_from_table.py results.txt

    # Option 3: Specify output image path
    python plot_from_table.py results.txt -o my_plot.png
"""

import argparse
import re
import sys


def parse_markdown_table(text: str):
    """
    Parse a Markdown table with format:
    | Block Size | H2D-naive | H2D-batch | H2D-swap | H2D-triton | D2H-naive | ... |
    | 256 B      | 1.2345    | 2.3456    | N/A      | ...        | ...       | ... |

    Returns:
        block_sizes: list of int (bytes)
        results: dict of {direction: {mode: [bandwidth_in_GiB_s]}}
    """
    lines = text.strip().split("\n")

    # Find table header line (contains "Block Size")
    header_idx = None
    for i, line in enumerate(lines):
        if "Block Size" in line and "|" in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find table header with 'Block Size'")

    # Parse header to get column order
    header_line = lines[header_idx]
    columns = [col.strip() for col in header_line.split("|")[1:-1]]
    # Expected: ["Block Size", "H2D-naive", "H2D-batch", ..., "D2H-naive", ...]

    # Parse data rows (skip header and separator line)
    block_sizes_bytes = []
    results = {}  # {direction: {mode: [bandwidths]}}

    # Initialize results structure
    for col in columns[1:]:  # Skip "Block Size"
        parts = col.split("-")
        if len(parts) >= 2:
            direction = parts[0]
            mode = "-".join(parts[1:])  # Handle modes like "triton"
            if direction not in results:
                results[direction] = {}
            if mode not in results[direction]:
                results[direction][mode] = []

    for line in lines[header_idx + 2:]:  # Skip header and separator
        line = line.strip()
        if not line.startswith("|") or "---" in line:
            continue

        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) != len(columns):
            continue

        # Parse block size string to bytes
        bs_str = cells[0]
        bs_bytes = parse_size_to_bytes(bs_str)
        if bs_bytes is None:
            continue
        block_sizes_bytes.append(bs_bytes)

        # Parse bandwidth values
        for col_idx, col_name in enumerate(columns[1:], start=1):
            parts = col_name.split("-")
            direction = parts[0]
            mode = "-".join(parts[1:])

            value_str = cells[col_idx]
            if value_str.upper() == "N/A" or value_str == "":
                results[direction][mode].append(None)
            else:
                try:
                    results[direction][mode].append(float(value_str))
                except ValueError:
                    results[direction][mode].append(None)

    return block_sizes_bytes, results


def parse_size_to_bytes(size_str: str) -> int | None:
    """
    Parse a human-readable size string like '256 B', '1 KiB', '2 MiB' to bytes.
    """
    size_str = size_str.strip()

    # Patterns: "123 B", "1.5 KiB", "2 MiB", "3 GiB", etc.
    match = re.match(r"([\d.]+)\s*(B|KiB|MiB|GiB|TiB|KB|MB|GB|TB)?$", size_str)
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2) or "B"

    # Binary units (used by format_size with use_binary=True)
    binary_multipliers = {
        "B": 1,
        "KiB": 1024,
        "MiB": 1024 ** 2,
        "GiB": 1024 ** 3,
        "TiB": 1024 ** 4,
    }

    # Decimal units (for safety, though benchmark uses binary)
    decimal_multipliers = {
        "B": 1,
        "KB": 1000,
        "MB": 1000 ** 2,
        "GB": 1000 ** 3,
        "TB": 1000 ** 4,
    }

    if unit in binary_multipliers:
        return int(value * binary_multipliers[unit])
    elif unit in decimal_multipliers:
        return int(value * decimal_multipliers[unit])

    return None


def format_size_bytes(num_bytes: int, decimal_places: int = 0) -> str:
    """Format byte count as human-readable string (binary units)."""
    if num_bytes == 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    base = 1024
    exponent = 0
    size = float(num_bytes)
    while size >= base and exponent < len(units) - 1:
        size /= base
        exponent += 1
    if decimal_places == 0:
        return f"{int(round(size))} {units[exponent]}"
    return f"{size:.{decimal_places}f} {units[exponent]}"


def plot_results(
        block_sizes: list,
        results: dict,
        output_file: str = None,
        title: str = "Block Copy Bandwidth",
):
    """
    Create a bandwidth plot (GiB/s vs block size in bytes).

    Args:
        block_sizes: List of block sizes in bytes.
        results: Nested dict: {direction: {mode: [bandwidth_GiB_s]}}.
        output_file: Path to save the plot (optional).
        title: Plot title.
    """
    import matplotlib.pyplot as plt

    directions = list(results.keys())
    modes = list(results[directions[0]].keys()) if directions else []

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    line_styles = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(figsize=(12, 8))

    for d_idx, direction in enumerate(directions):
        for m_idx, mode in enumerate(modes):
            bw_list = results.get(direction, {}).get(mode, [])
            if not bw_list:
                continue

            x_vals, y_vals = [], []
            for bs, bw in zip(block_sizes, bw_list):
                if bw is not None:
                    x_vals.append(bs)
                    y_vals.append(bw)  # Already in GiB/s

            if not x_vals:
                continue

            label = f"{direction}-{mode}"
            color = colors[(d_idx * len(modes) + m_idx) % len(colors)]
            style = line_styles[m_idx % len(line_styles)]

            ax.plot(
                x_vals,
                y_vals,
                label=label,
                color=color,
                linestyle=style,
                marker="o",
                markersize=6,
                linewidth=1.5,
                alpha=0.85,
            )

    # Configure X-axis (log2 scale)
    ax.set_xscale("log", base=2)
    ax.set_xticks(block_sizes)
    labels = [format_size_bytes(bs, decimal_places=0) for bs in block_sizes]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Block Size", fontsize=12)

    # Configure Y-axis
    ax.set_yscale("linear")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(
        plt.ScalarFormatter(useOffset=False, useMathText=False)
    )
    ax.set_ylabel("Bandwidth (GiB/s)", fontsize=12)

    ax.set_title(title, fontsize=14)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(
        bbox_to_anchor=(1.04, 1),
        loc="upper left",
        framealpha=0.9,
        fontsize=9,
    )

    fig.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot bandwidth results from vLLM benchmark table output."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input file containing the benchmark output. "
             "If not provided, reads from stdin.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image file path (e.g., plot.png). "
             "If not provided, displays the plot interactively.",
    )
    parser.add_argument(
        "--title",
        default="Block Copy Bandwidth (GiB/s)",
        help="Plot title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for output image. Default: 150.",
    )
    args = parser.parse_args()

    # Read input
    if args.input:
        with open(args.input, "r") as f:
            text = f.read()
    else:
        print("Reading from stdin... (Ctrl+D to end input)", file=sys.stderr)
        text = sys.stdin.read()

    if not text.strip():
        print("Error: No input provided.", file=sys.stderr)
        sys.exit(1)

    # Parse table
    try:
        block_sizes, results = parse_markdown_table(text)
    except ValueError as e:
        print(f"Error parsing table: {e}", file=sys.stderr)
        print("Input text:\n" + text[:500], file=sys.stderr)
        sys.exit(1)

    if not block_sizes:
        print("Error: No data rows found in table.", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(block_sizes)} block sizes, "
          f"{len(results)} directions, "
          f"{len(results[list(results.keys())[0]])} modes.")

    # Plot
    plot_results(
        block_sizes,
        results,
        output_file=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()