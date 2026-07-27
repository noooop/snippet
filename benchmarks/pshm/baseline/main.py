#!/usr/bin/env python3
"""
Unified CPU/GPU block copy bandwidth benchmark.

Tests (selectable via --tests):
  1. H2H          – ordinary CPU memcpy (Python loop)
  2. H2H (shm)    – CPU copy via shared memory
  3. H2D (batch)  – vLLM swap_blocks_batch (host → device)
  4. D2H (batch)  – vLLM swap_blocks_batch (device → host)
  5. D2D (naive)  – device-to-device copy using PyTorch tensor indexing
"""

import argparse
import contextlib
import multiprocessing as mp
import platform
import random
import time
from multiprocessing import shared_memory
from unittest.mock import patch

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Optional dependency checks
# ---------------------------------------------------------------------------
try:
    from vllm import _custom_ops as ops

    HAS_VLLM_OPS = True
except ImportError:
    HAS_VLLM_OPS = False
    print(
        "Warning: vllm._custom_ops.swap_blocks_batch not available; "
        "H2D(batch) and D2H(batch) will be skipped."
    )

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available – plot will be skipped.")


# ---------------------------------------------------------------------------
# Formatting helpers
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
# System information
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
# Shared memory helpers
# ---------------------------------------------------------------------------
class SharedMemoryTensor:
    """Manages a single shared memory segment wrapped as a torch tensor."""

    def __init__(self, size: int):
        self.size = size
        self._shm = None
        self._process = None
        self._stop_event = mp.Event()
        self._tensor = None

    def __enter__(self):
        """Create the shared memory segment in a worker process and map it."""
        parent_conn, child_conn = mp.Pipe()
        self._process = mp.Process(
            target=self._worker, args=(child_conn, self._stop_event)
        )
        self._process.start()
        shm_name = parent_conn.recv()
        parent_conn.close()

        # Avoid double registration in resource tracker
        with patch(
            "multiprocessing.resource_tracker.register", lambda *args, **kwargs: None
        ):
            self._shm = shared_memory.SharedMemory(name=shm_name)

        np_dtype = np.dtype(np.uint8)
        arr = np.ndarray(self.size, dtype=np_dtype, buffer=self._shm.buf)
        self._tensor = torch.from_numpy(arr)
        return self._tensor

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Signal the worker and clean up resources."""
        del self._tensor
        self._shm.close()
        self._stop_event.set()
        self._process.join()
        self._shm.unlink()
        return False

    @staticmethod
    def _worker(conn, stop_event):
        """Worker that creates a shared memory segment and waits for stop."""
        conn.recv()
        pass


@contextlib.contextmanager
def shared_tensor_pair(total_bytes: int):
    """Context manager yielding (src, dst) tensors backed by shared memory."""

    def _create_shm_tensor(size):
        parent_conn, child_conn = mp.Pipe()
        stop_event = mp.Event()
        process = mp.Process(target=_shm_worker, args=(size, child_conn, stop_event))
        process.start()
        shm_name = parent_conn.recv()
        parent_conn.close()
        with patch(
            "multiprocessing.resource_tracker.register", lambda *args, **kwargs: None
        ):
            shm = shared_memory.SharedMemory(name=shm_name)
        tensor = torch.from_numpy(np.ndarray(size, dtype=np.uint8, buffer=shm.buf))
        return shm, tensor, process, stop_event

    shm_src, src_t, proc_src, stop_src = _create_shm_tensor(total_bytes)
    shm_dst, dst_t, proc_dst, stop_dst = _create_shm_tensor(total_bytes)
    try:
        yield src_t, dst_t
    finally:
        del src_t
        shm_src.close()
        stop_src.set()
        proc_src.join()

        del dst_t
        shm_dst.close()
        stop_dst.set()
        proc_dst.join()


def _shm_worker(size: int, conn: mp.connection.Connection, stop_event: mp.Event):
    """Creates a shared memory segment, sends its name, then waits for stop."""
    shm = shared_memory.SharedMemory(size=size, create=True)
    try:
        conn.send(shm.name)
        conn.close()
        stop_event.wait()
    finally:
        shm.close()
        shm.unlink()


# ---------------------------------------------------------------------------
# Index generation
# ---------------------------------------------------------------------------
def generate_random_indices(num_blocks: int, n_iters: int) -> tuple:
    """Return a pair of numpy arrays (src_indices, dst_indices) of shape (n_iters,)."""
    tasks = [
        (random.randint(0, num_blocks - 1), random.randint(0, num_blocks - 1))
        for _ in range(n_iters)
    ]
    src_idx = np.array([t[0] for t in tasks], dtype=np.intp)
    dst_idx = np.array([t[1] for t in tasks], dtype=np.intp)
    return src_idx, dst_idx


# ---------------------------------------------------------------------------
# Benchmark implementations
# ---------------------------------------------------------------------------
def benchmark_h2h(total_bytes: int, block_sizes: list, n_iters: int):
    """CPU memcpy (H2H) using Python loop indexing."""
    print("\n=== H2H (memcpy) ===")
    dtype = torch.uint8
    src = torch.randn(total_bytes // 4, dtype=torch.float32, device="cpu").view(dtype)
    dst = torch.randn(total_bytes // 4, dtype=torch.float32, device="cpu").view(dtype)
    print(f"Allocated {format_size(src.nelement() * src.element_size())} for H2H")

    bandwidths = []
    for bs_bytes in block_sizes:
        num_blocks = total_bytes // bs_bytes
        src_indices, dst_indices = generate_random_indices(num_blocks, n_iters)
        s_view = src.view(-1, bs_bytes)
        d_view = dst.view(-1, bs_bytes)

        # Warm-up
        for i, j in zip(src_indices, dst_indices):
            d_view[i] = s_view[j]

        # Timed run
        src_indices, dst_indices = generate_random_indices(num_blocks, n_iters)
        start = time.perf_counter()
        for i, j in zip(src_indices, dst_indices):
            d_view[i] = s_view[j]
        elapsed = time.perf_counter() - start

        bw = (bs_bytes * n_iters) / elapsed
        bandwidths.append(bw)
        print(
            f"  block size: {format_size(bs_bytes):>12s}  bandwidth: {format_bandwidth(bw)}"
        )

    return bandwidths


def benchmark_h2h_shm(total_bytes: int, block_sizes: list, n_iters: int):
    """CPU shared memory copy (H2H) using Python loop indexing."""
    print("\n=== H2H (shm) ===")
    bandwidths = []
    with shared_tensor_pair(total_bytes) as (src, dst):
        # Fill with random data
        src[:] = torch.randn(total_bytes // 4, dtype=torch.float32).view(torch.uint8)
        dst[:] = torch.randn(total_bytes // 4, dtype=torch.float32).view(torch.uint8)
        print(f"Allocated {format_size(total_bytes)} for shm")

        for bs_bytes in block_sizes:
            num_blocks = total_bytes // bs_bytes
            src_indices, dst_indices = generate_random_indices(num_blocks, n_iters)
            s_view = src.view(-1, bs_bytes)
            d_view = dst.view(-1, bs_bytes)

            # Warm-up
            for i, j in zip(src_indices, dst_indices):
                d_view[i] = s_view[j]

            # Timed run
            src_indices, dst_indices = generate_random_indices(num_blocks, n_iters)
            start = time.perf_counter()
            for i, j in zip(src_indices, dst_indices):
                d_view[i] = s_view[j]
            elapsed = time.perf_counter() - start

            bw = (bs_bytes * n_iters) / elapsed
            bandwidths.append(bw)
            print(
                f"  block size: {format_size(bs_bytes):>12s}  bandwidth: {format_bandwidth(bw)}"
            )

    return bandwidths


def benchmark_gpu(
    host: torch.Tensor,
    device: torch.Tensor,
    block_sizes: list,
    n_iters: int,
    direction: str,
    mode: str,
):
    """
    Benchmark GPU block copies.

    Args:
        host: Pinned CPU tensor (required for H2D/D2H directions).
        device: GPU tensor.
        block_sizes: List of block sizes in bytes.
        n_iters: Number of timed iterations.
        direction: "H2D", "D2H", or "D2D".
        mode: "batch" (uses vLLM swap_blocks_batch) or "naive" (Python loop).
    """
    if not torch.cuda.is_available():
        print(f"CUDA not available, skipping {direction}-{mode}.")
        return [None] * len(block_sizes)

    if mode == "batch" and not HAS_VLLM_OPS:
        print(f"vLLM ops unavailable, skipping {direction}-{mode}.")
        return [None] * len(block_sizes)

    label = f"{direction} ({mode})"
    print(f"\n=== {label} ===")

    bandwidths = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        for bs_bytes in block_sizes:
            if direction in ("H2D", "D2H"):
                host_view = host.view(-1, bs_bytes)
                device_view = device.view(-1, bs_bytes)
            else:  # D2D
                device_view = device.view(-1, bs_bytes)

            num_blocks = device_view.size(0)

            # Pre-generate random (src_idx, dst_idx) pairs
            tasks = [
                (random.randint(0, num_blocks - 1), random.randint(0, num_blocks - 1))
                for _ in range(n_iters)
            ]

            if mode == "batch":
                if direction == "H2D":
                    src_addrs = torch.tensor(
                        [host_view[i].data_ptr() for i, j in tasks], dtype=torch.int64
                    )
                    dst_addrs = torch.tensor(
                        [device_view[j].data_ptr() for i, j in tasks], dtype=torch.int64
                    )
                else:  # D2H
                    src_addrs = torch.tensor(
                        [device_view[i].data_ptr() for i, j in tasks], dtype=torch.int64
                    )
                    dst_addrs = torch.tensor(
                        [host_view[j].data_ptr() for i, j in tasks], dtype=torch.int64
                    )
                sizes = torch.full((n_iters,), bs_bytes, dtype=torch.int64)

                def run_batch(iters):
                    stream = torch.cuda.Stream()
                    with torch.cuda.stream(stream):
                        ops.swap_blocks_batch(src_addrs, dst_addrs, sizes)
                    torch.cuda.current_stream().wait_stream(stream)

                copy_func = run_batch

            elif mode == "naive":
                if direction == "D2D":

                    def run_naive(iters):
                        for k in range(iters):
                            i, j = tasks[k]
                            device_view[j] = device_view[i]
                else:
                    raise NotImplementedError("Naive mode only implemented for D2D")
                copy_func = run_naive

            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Warm-up
            copy_func(1)
            torch.cuda.synchronize()

            # Timed run
            start_event.record()
            copy_func(n_iters)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            elapsed_s = elapsed_ms / 1000.0

            bw = (bs_bytes * n_iters) / elapsed_s
            bandwidths.append(bw)
            print(
                f"  block size: {format_size(bs_bytes):>12s}  bandwidth: {format_bandwidth(bw)}"
            )

    return bandwidths


# ---------------------------------------------------------------------------
# Output helpers
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
    fig, ax = plt.subplots(figsize=(12, 8))
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
    ax.set_title(f"CPU/GPU Block Copy Bandwidth\n({sys_info})")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ALL_TESTS = ["H2H", "H2H (shm)", "H2D (batch)", "D2H (batch)", "D2D (naive)"]


def main():
    parser = argparse.ArgumentParser(
        description="Unified CPU/GPU block copy bandwidth benchmark."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2**32,
        help="Total memory allocated in bytes (must be divisible by block sizes). "
        "Default: 2**32 (4 GiB).",
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
        "--tests",
        nargs="+",
        choices=ALL_TESTS,
        default=["H2H", "H2H (shm)", "H2D (batch)", "D2H (batch)"],
        help="Which tests to run. Default: H2H, H2H (shm), H2D (batch), D2H (batch). "
        "D2D (naive) is disabled by default.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not display or save the bandwidth plot.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="block_copy_baseline.png",
        help="Save the plot to the given file path.",
    )
    args = parser.parse_args()

    total_bytes = args.size
    n_iters = args.n_iters
    block_sizes = [2**n for n in range(args.min_block_exp, args.max_block_exp + 1)]

    # Ensure total_bytes is a multiple of the largest block size
    max_block = block_sizes[-1]
    if total_bytes % max_block != 0:
        print(
            f"Warning: total_bytes ({total_bytes}) is not a multiple of the largest "
            f"block size ({max_block}). Adjusting total_bytes to the next multiple."
        )
        total_bytes = ((total_bytes // max_block) + 1) * max_block
        print(f"New total_bytes: {total_bytes}")

    results = {}

    # GPU tensors (only allocated if any GPU test is selected)
    host_raw = device_raw = None
    need_gpu = any(
        t in args.tests for t in ["H2D (batch)", "D2H (batch)", "D2D (naive)"]
    )
    if need_gpu:
        if not torch.cuda.is_available():
            print("CUDA not available; all GPU tests will be skipped.")
        else:
            dtype = torch.uint8
            host_raw = torch.randn(
                total_bytes // 4, dtype=torch.float32, device="cpu"
            ).view(dtype)
            device_raw = torch.randn(
                total_bytes // 4, dtype=torch.float32, device="cuda"
            ).view(dtype)
            # Pin host memory for faster H2D/D2H transfers
            try:
                from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor

                pin_tensor(host_raw)
            except ImportError:
                host_raw = host_raw.pin_memory()
            print(f"\nAllocated {format_size(total_bytes)} for host & device tensors")

    # Execute selected tests in order
    for test_name in args.tests:
        if test_name == "H2H":
            results[test_name] = benchmark_h2h(total_bytes, block_sizes, n_iters)
        elif test_name == "H2H (shm)":
            results[test_name] = benchmark_h2h_shm(total_bytes, block_sizes, n_iters)
        elif test_name == "H2D (batch)":
            if host_raw is not None and device_raw is not None:
                results[test_name] = benchmark_gpu(
                    host_raw,
                    device_raw,
                    block_sizes,
                    n_iters,
                    direction="H2D",
                    mode="batch",
                )
            else:
                results[test_name] = [None] * len(block_sizes)
        elif test_name == "D2H (batch)":
            if host_raw is not None and device_raw is not None:
                results[test_name] = benchmark_gpu(
                    host_raw,
                    device_raw,
                    block_sizes,
                    n_iters,
                    direction="D2H",
                    mode="batch",
                )
            else:
                results[test_name] = [None] * len(block_sizes)
        elif test_name == "D2D (naive)":
            if host_raw is not None and device_raw is not None:
                results[test_name] = benchmark_gpu(
                    host_raw,
                    device_raw,
                    block_sizes,
                    n_iters,
                    direction="D2D",
                    mode="naive",
                )
            else:
                results[test_name] = [None] * len(block_sizes)

    # Print results and plot
    if results:
        print_results_table(block_sizes, results)

    if (not args.no_plot or args.save_plot) and results:
        plot_results(block_sizes, results, output_file=args.save_plot)


if __name__ == "__main__":
    main()
