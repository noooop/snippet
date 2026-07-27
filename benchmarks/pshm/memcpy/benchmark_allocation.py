#!/usr/bin/env python3
"""
Benchmark CPU block copy bandwidth for four allocation/access patterns:
  1. random_read    – random read from a big buffer into a fixed target
  2. random_write   – read from a fixed source into random positions of a big buffer
  3. random_rw      – random read from one big buffer into random positions of another (static)
  4. dynamic        – read from a fixed source, allocate a new tensor per block (dynamic)

Results are printed as a Markdown table and a bandwidth plot is generated.
"""

import argparse
import platform
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# matplotlib check
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available – plot will be skipped.")


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------
def format_size(
    num_bytes: int,
    decimal_places: int = 4,
    use_binary: bool = True,
    target_unit: str = None,
) -> str:
    """Format a byte count as a human-readable string."""
    if num_bytes == 0:
        return f"0 {target_unit or 'B'}"
    units = ["B", "KiB", "MiB", "GiB"] if use_binary else ["B", "KB", "MB", "GB"]
    base = 1024 if use_binary else 1000
    if target_unit is not None:
        target_exp = units.index(target_unit)
        size = num_bytes / (base**target_exp)
        return f"{size:.{decimal_places}f} {target_unit}"
    exponent = 0
    size = num_bytes
    while size >= base and exponent < len(units) - 1:
        size /= base
        exponent += 1
    return f"{size:.{decimal_places}f} {units[exponent]}"


def format_bandwidth(bytes_per_sec: float, decimal_places: int = 4) -> str:
    """Format bandwidth in GiB/s."""
    return (
        format_size(
            int(bytes_per_sec), decimal_places=decimal_places, target_unit="GiB"
        )
        + "/s"
    )


# ---------------------------------------------------------------------------
# System info helper
# ---------------------------------------------------------------------------
def get_system_info() -> str:
    """Return a string describing CPU model and total memory."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        cpu_model = platform.machine()

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    mem_kb = int(parts[1])
                    mem_str = f"{mem_kb / (1024**2):.1f} GiB"
                    break
    except Exception:
        mem_str = "unknown"

    return f"CPU: {cpu_model}, Memory: {mem_str}"


# ---------------------------------------------------------------------------
# core measurement function
# ---------------------------------------------------------------------------
def measure_bandwidth(
    copy_func,
    total_bytes: int,
    block_sizes: list,
    n_iters: int,
    label: str = "",
    indices_generator=None,
) -> list:
    """
    Measure bandwidth for a given copy_func over a range of block sizes.
    copy_func(src_indices, dst_indices, block_bytes) performs the actual copy.
    indices_generator(num_blocks, n_iters) returns (src_indices, dst_indices).
    Returns a list of bandwidths in bytes/s, one per block_size.
    """
    if indices_generator is None:

        def default_gen(num_blocks, n_iters):
            return generate_random_indices(num_blocks, n_iters)

        indices_generator = default_gen

    bandwidths = []
    for bs_bytes in block_sizes:
        num_blocks = total_bytes // bs_bytes

        # Warm-up
        src_indices, dst_indices = indices_generator(num_blocks, n_iters)
        copy_func(src_indices, dst_indices, bs_bytes)

        # Timed run
        src_indices, dst_indices = indices_generator(num_blocks, n_iters)
        start = time.perf_counter()
        copy_func(src_indices, dst_indices, bs_bytes)
        elapsed = time.perf_counter() - start

        bw = (bs_bytes * n_iters) / elapsed
        bandwidths.append(bw)
        print(
            f"[{label}] size: {format_size(bs_bytes)}, Bandwidth: {format_bandwidth(bw)}"
        )

    return bandwidths


def generate_random_indices(num_blocks: int, n_iters: int) -> tuple:
    """Return random src and dst indices."""
    src = np.random.randint(0, num_blocks, size=n_iters, dtype=np.intp)
    dst = np.random.randint(0, num_blocks, size=n_iters, dtype=np.intp)
    return src, dst


# ---------------------------------------------------------------------------
# benchmark scenario constructors
# ---------------------------------------------------------------------------
def run_random_read(total_bytes, block_sizes, n_iters):
    """Test 1: random read from big source to fixed target."""
    dtype = torch.uint8
    max_block = max(block_sizes)
    src_big = torch.randn(total_bytes // 4, dtype=torch.float32).view(dtype)
    fixed_dst = torch.zeros(
        max_block, dtype=dtype
    )  # fixed target buffer (size max_block)
    print(
        f"Allocated {format_size(total_bytes)} for src_big and {format_size(max_block)} for fixed_dst (random_read)"
    )

    def indices_gen(num_blocks, n_iters):
        src_idx = np.random.randint(0, num_blocks, size=n_iters, dtype=np.intp)
        dst_idx = np.zeros(n_iters, dtype=np.intp)  # always write to block 0
        return src_idx, dst_idx

    def copy_func(src_indices, dst_indices, block_bytes):
        s_view = src_big.view(-1, block_bytes)
        d_view = fixed_dst.view(-1, block_bytes)  # shape (1, block_bytes)
        for i, j in zip(src_indices, dst_indices):
            d_view[j] = s_view[i]  # j is always 0

    with torch.inference_mode():
        return measure_bandwidth(
            copy_func,
            total_bytes,
            block_sizes,
            n_iters,
            "random_read",
            indices_generator=indices_gen,
        )


def run_random_write(total_bytes, block_sizes, n_iters):
    """Test 2: random write from fixed source to big target."""
    dtype = torch.uint8
    max_block = max(block_sizes)
    dst_big = torch.randn(total_bytes // 4, dtype=torch.float32).view(dtype)
    fixed_src = torch.zeros(max_block, dtype=dtype)  # fixed source buffer
    # Fill fixed_src with some data (e.g., random) so reads are meaningful
    fixed_src[:] = torch.randint(0, 256, (max_block,), dtype=dtype)
    print(
        f"Allocated {format_size(total_bytes)} for dst_big and {format_size(max_block)} for fixed_src (random_write)"
    )

    def indices_gen(num_blocks, n_iters):
        src_idx = np.zeros(n_iters, dtype=np.intp)  # always read from block 0
        dst_idx = np.random.randint(0, num_blocks, size=n_iters, dtype=np.intp)
        return src_idx, dst_idx

    def copy_func(src_indices, dst_indices, block_bytes):
        s_view = fixed_src.view(-1, block_bytes)  # shape (1, block_bytes)
        d_view = dst_big.view(-1, block_bytes)
        for i, j in zip(src_indices, dst_indices):
            d_view[j] = s_view[i]  # i is always 0

    with torch.inference_mode():
        return measure_bandwidth(
            copy_func,
            total_bytes,
            block_sizes,
            n_iters,
            "random_write",
            indices_generator=indices_gen,
        )


def run_random_rw(total_bytes, block_sizes, n_iters):
    """Test 3: random read and random write (original memcpy style)."""
    dtype = torch.uint8
    src_big = torch.randn(total_bytes // 4, dtype=torch.float32).view(dtype)
    dst_big = torch.randn(total_bytes // 4, dtype=torch.float32).view(dtype)
    print(f"Allocated {format_size(total_bytes)} for src_big and dst_big (random_rw)")

    def copy_func(src_indices, dst_indices, block_bytes):
        s_view = src_big.view(-1, block_bytes)
        d_view = dst_big.view(-1, block_bytes)
        for i, j in zip(src_indices, dst_indices):
            d_view[j] = s_view[i]

    with torch.inference_mode():
        return measure_bandwidth(
            copy_func, total_bytes, block_sizes, n_iters, "random_rw"
        )


def run_dynamic(total_bytes, block_sizes, n_iters):
    """Test 4: read from fixed source, allocate new tensor per block (dynamic allocation)."""
    dtype = torch.uint8
    max_block = max(block_sizes)
    fixed_src = torch.randint(0, 256, (max_block,), dtype=dtype)  # fixed source
    print(f"Allocated {format_size(max_block)} for fixed_src (dynamic)")

    def indices_gen(num_blocks, n_iters):
        src_idx = np.zeros(n_iters, dtype=np.intp)  # always read from block 0
        dst_idx = np.zeros(n_iters, dtype=np.intp)  # not used
        return src_idx, dst_idx

    def copy_func(src_indices, dst_indices, block_bytes):
        s_view = fixed_src.view(-1, block_bytes)
        for idx in src_indices:
            tensor_copy = s_view[idx].clone()
            assert s_view.numpy().ctypes.data != tensor_copy.numpy().ctypes.data

    with torch.inference_mode():
        return measure_bandwidth(
            copy_func,
            total_bytes,
            block_sizes,
            n_iters,
            "dynamic",
            indices_generator=indices_gen,
        )


# ---------------------------------------------------------------------------
# output functions
# ---------------------------------------------------------------------------
def print_results_table(block_sizes: list, results: dict):
    """
    Print a Markdown table of bandwidths (GiB/s).
    System information is printed before the table.
    """
    sys_info = get_system_info()
    print(f"\n**System:** {sys_info}\n")

    bench_names = list(results.keys())
    header = ["Block Size"] + bench_names
    print("### Bandwidth Results (GiB/s)")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join([" --- " for _ in header]) + "|")

    for i, bs in enumerate(block_sizes):
        row = [format_size(bs, decimal_places=0)]
        for name in bench_names:
            bw = results[name][i]
            if bw is None:
                row.append("N/A")
            else:
                gib_s = bw / (1024**3)
                row.append(f"{gib_s:.4f}")
        print("| " + " | ".join(row) + " |")


def plot_results(block_sizes: list, results: dict, output_file: str = None):
    """
    Bandwidth plot (GiB/s vs block size).
    X‑axis: log2, human‑readable ticks.
    Y‑axis: linear, starting at 0.
    System information is included in the plot title.
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plot: matplotlib not available.")
        return

    sys_info = get_system_info()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (name, bw_list) in enumerate(results.items()):
        x_vals, y_vals = [], []
        for bs, bw in zip(block_sizes, bw_list):
            if bw is not None:
                x_vals.append(bs)
                y_vals.append(bw / (1024**3))
        if not x_vals:
            continue
        ax.plot(
            x_vals,
            y_vals,
            label=name,
            color=colors[idx % len(colors)],
            marker="o",
            markersize=4,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(block_sizes)
    ax.set_xticklabels(
        [format_size(bs, decimal_places=0) for bs in block_sizes],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("Block Size")
    ax.set_ylabel("Bandwidth (GiB/s)")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
    ax.set_title(f"CPU Block Copy Bandwidth\n({sys_info})")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark static vs dynamic memory allocation for CPU block copy."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2**32,
        help="Total memory allocated in bytes (must be divisible by block sizes). Default: 2**32 (4 GiB).",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=100,
        help="Number of timed iterations per block size. Default: 100.",
    )
    parser.add_argument(
        "--min-block-exp",
        type=int,
        default=8,
        help="Exponent for the smallest block size (2**exp bytes). Default: 8 (256 B).",
    )
    parser.add_argument(
        "--max-block-exp",
        type=int,
        default=30,
        help="Exponent for the largest block size (2**exp bytes). Default: 30 (1 GiB).",
    )
    parser.add_argument(
        "--bench",
        nargs="+",
        choices=["random_read", "random_write", "random_rw", "dynamic"],
        default=["random_read", "random_write", "random_rw", "dynamic"],
        help="Which benchmarks to run. Default: all four.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not display or save the bandwidth plot.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="benchmark_cpu_block_copy.png",
        help="Save the plot to the given file path.",
    )
    args = parser.parse_args()

    total_bytes = args.size
    n_iters = args.n_iters
    block_sizes = [2**n for n in range(args.min_block_exp, args.max_block_exp + 1)]

    max_block = block_sizes[-1]
    if total_bytes % max_block != 0:
        print(
            f"Warning: total_bytes ({total_bytes}) is not a multiple of the largest "
            f"block size ({max_block}). Adjusting total_bytes to the next multiple."
        )
        total_bytes = ((total_bytes // max_block) + 1) * max_block
        print(f"New total_bytes: {total_bytes}")

    # Map benchmark names to functions
    benchmarks = {
        "random_read": run_random_read,
        "random_write": run_random_write,
        "random_rw": run_random_rw,
        "dynamic": run_dynamic,
    }

    results = {}
    for name in args.bench:
        if name in benchmarks:
            print(f"\n=== Benchmark: {name} ===")
            results[name] = benchmarks[name](total_bytes, block_sizes, n_iters)

    if results:
        print_results_table(block_sizes, results)

    if (not args.no_plot or args.save_plot) and results:
        plot_results(block_sizes, results, output_file=args.save_plot)


if __name__ == "__main__":
    main()
